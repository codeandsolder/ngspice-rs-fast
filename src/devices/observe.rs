//! Device observations from the same physical loads used by the solver.
//!
//! C: `frontend/outitf.c`, `spiceif.c::if_getparam`, device `*ask.c` routines.
//! Loads are disposable; no observation accepts state or advances history.
use std::collections::{BTreeMap, BTreeSet};

use crate::devices::circuit::{PreparedMutualTerms, PreparedPoint, TopologyId};
use crate::devices::state::IsolatedState;
use crate::devices::traits::{IsolatedRhs, StampRhs};
use crate::devices::{
    Circuit, Device, Forcing, LoadRequest, ModelContext, PreparedRun, TrialState,
};
use crate::maths::{SparseMatrix, Vector};
use crate::primitives::{Complex, NodeId, Real, SpiceError, SpiceResult};

fn gap(name: &str) -> SpiceError {
    SpiceError::not_yet_ported(
        format!("device observation {name}"),
        "src/frontend/spiceif.c (if_getparam), src/spicelib/devices/*/*ask.c",
    )
}

fn query(name: &str) -> SpiceResult<(&str, &str)> {
    if let Some(device) = name.strip_prefix("i(").and_then(|s| s.strip_suffix(')')) {
        return Ok((device, "i"));
    }
    let (device, parameter) = name
        .strip_prefix('@')
        .and_then(|s| s.split_once('['))
        .ok_or_else(|| gap(name))?;
    let parameter = parameter
        .strip_suffix(']')
        .filter(|s| !s.is_empty() && !s.contains(['[', ']']))
        .ok_or_else(|| gap(name))?;
    if device.is_empty() {
        return Err(gap(name));
    }
    Ok((device, parameter))
}

fn terminal(device: &dyn Device, key: &str) -> Option<usize> {
    // JFET (jfetask.c) and MESFET (mesask.c) port labels use the same
    // terminal numbering as their corresponding device load.
    match (device.designator(), key) {
        (_, "i" | "current") | ('d' | 'm' | 'j', "id") | ('q', "ic") | ('z', "cd") => Some(0),
        ('q', "ib") | ('m' | 'j', "ig") | ('z', "cg") => Some(1),
        ('q', "ie") | ('m' | 'j' | 'z', "is") => Some(2),
        ('m', "ib") => Some(3),
        _ => None,
    }
}

fn unit(device: char, key: &str) -> &'static str {
    if key == "ic" && device == 'c' {
        return "voltage";
    }
    // mesask.c: drain, gate and gate-drain currents.
    if matches!(key, "cd" | "cg" | "cgd") && device == 'z' {
        return "current";
    }
    match key {
        "i" | "current" | "id" | "ic" | "ib" | "ie" | "is" | "ig" | "igd" => "current",
        "r" | "resistance" => "resistance",
        "conductance" | "g" => "conductance",
        "c" | "cap" | "capacitance" => "capacitance",
        "l" | "inductance" => "inductance",
        "temp" | "dtemp" => "temperature",
        "p" | "power" => "power",
        _ => "parameter",
    }
}

/// A column bound to finalized device identity and a typed ask selector.
#[derive(Debug)]
pub struct Observation {
    name: String,
    unit: &'static str,
    device: usize,
    selector: Selector,
}

impl Observation {
    /// Original (canonical lower-case) plot column name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Rawfile unit derived once from the resolved device and quantity.
    #[must_use]
    pub const fn unit(&self) -> &'static str {
        self.unit
    }
}

#[derive(Debug)]
enum Selector {
    Current {
        load: usize,
        slot: usize,
        signed: bool,
    },
    Power {
        load: usize,
        slot: usize,
        rows: [Option<usize>; 2],
    },
    SourceDc,
    // Device-specific ask keywords are resolved by the device against the
    // effective context; their spelling is never parsed during evaluation.
    Parameter(Box<str>),
}

#[derive(Default)]
pub(super) struct ObservationPlan {
    topology: Option<TopologyId>,
    queries: Vec<Observation>,
    loads: Vec<TerminalLayout>,
}

/// Physical terminal rows in first-port order, with a separate lookup index
/// for each unfolded matrix contribution. Aliased ports share one slot;
/// the final slot is ground's negative sum of distinct physical rows.
struct TerminalLayout {
    device: usize,
    rows: Vec<usize>,
    by_row: Vec<(usize, usize)>,
    ports: Vec<usize>,
}

impl TerminalLayout {
    fn new(circuit: &Circuit, device: usize, terminals: &[NodeId]) -> Self {
        let mut rows = Vec::new();
        let mut by_row = BTreeMap::new();
        let mut ports = Vec::with_capacity(terminals.len());
        for &node in terminals {
            ports.push(circuit.unknowns().node_row(node).map(|row| {
                *by_row.entry(row).or_insert_with(|| {
                    let slot = rows.len();
                    rows.push(row);
                    slot
                })
            }));
        }
        Self {
            device,
            ports: ports
                .into_iter()
                .map(|slot| slot.unwrap_or(rows.len()))
                .collect(),
            rows,
            by_row: by_row.into_iter().collect(),
        }
    }

    fn residuals(
        &self,
        matrix: &SparseMatrix,
        rhs: impl Fn(usize) -> Real,
        solution: &Vector,
        currents: &mut [Real],
    ) {
        currents.fill(0.);
        for triplet in matrix.triplets() {
            if let Ok(index) = self
                .by_row
                .binary_search_by_key(&triplet.row, |&(row, _)| row)
            {
                currents[self.by_row[index].1] += triplet.value * solution.as_slice()[triplet.col];
            }
        }
        let mut ground = 0.;
        for (&row, current) in self.rows.iter().zip(&mut *currents) {
            *current -= rhs(row);
            ground -= *current;
        }
        currents[self.rows.len()] = ground;
    }
}

struct LoadedTerminals {
    ready: bool,
    currents: Vec<Real>,
}

/// Capacity is analysis-local; no previous point's currents or trial
/// state are considered valid after the next call begins.
pub(super) struct ObservationWorkspace {
    matrix: SparseMatrix,
    rhs: IsolatedRhs,
    state: IsolatedState,
    loaded: Vec<LoadedTerminals>,
}

impl ObservationWorkspace {
    fn new(circuit: &Circuit, plan: &ObservationPlan) -> Self {
        let n = circuit.unknown_count();
        Self {
            matrix: SparseMatrix::new(n, n),
            rhs: IsolatedRhs::new(n),
            state: IsolatedState::default(),
            loaded: plan
                .loads
                .iter()
                .map(|layout| LoadedTerminals {
                    ready: false,
                    currents: vec![0.; layout.rows.len() + 1],
                })
                .collect(),
        }
    }

    fn begin(&mut self) {
        for load in &mut self.loaded {
            load.ready = false;
        }
    }
}

/// Read-only point loads. The effective replacements and mutual terms are
/// supplied by the same prepared context that the solver used.
struct PointLoads<'a, 'b> {
    circuit: &'a Circuit,
    request: &'a LoadRequest<'b>,
    sources: &'a [(&'a str, Real)],
    time: Option<(Real, Forcing)>,
    previous: Option<&'a TrialState>,
    point: &'a PreparedPoint,
    mutual: Option<PreparedMutualTerms>,
    workspace: &'a mut ObservationWorkspace,
}

impl PointLoads<'_, '_> {
    fn current(&mut self, load: usize, slot: usize, name: &str) -> SpiceResult<Real> {
        let layout = &self.circuit.runtime.observations.loads[load];
        let work = &mut self.workspace;
        let loaded = &mut work.loaded[load];
        if !loaded.ready {
            if self.mutual.is_none() {
                self.mutual = Some(self.point.mutual(self.circuit)?);
            }
            let device = self.point.device(self.circuit, layout.device);
            work.matrix.clear();
            work.rhs.clear();
            let states = self.request.history.isolated_device(
                &mut work.state,
                self.previous,
                layout.device,
            )?;
            let stamped = self.circuit.stamp_isolated(
                layout.device,
                device,
                self.request,
                &mut work.matrix,
                StampRhs::isolated(&mut work.rhs),
                states,
                self.mutual.as_deref().unwrap_or(&[]),
            );
            work.state.clear();
            stamped?;
            if device.designator() == 'i' {
                let value = device
                    .observation_source_current(self.time)?
                    .ok_or_else(|| gap(name))?;
                let value = self
                    .sources
                    .iter()
                    .find(|(name, _)| name.eq_ignore_ascii_case(device.name()))
                    .map_or(value, |(_, v)| *v * device.observation_source_multiplier());
                for (terminal, sign) in device.terminals().iter().take(2).zip([-1., 1.]) {
                    if let Some(row) = self.circuit.unknowns().node_row(*terminal) {
                        work.rhs.set(row, sign * value)?;
                    }
                }
            }
            layout.residuals(
                &work.matrix,
                |row| work.rhs.get(row),
                self.request.solution,
                &mut loaded.currents,
            );
            loaded.ready = true;
        }
        loaded.currents.get(slot).copied().ok_or_else(|| gap(name))
    }

    fn observe(&mut self, query: &Observation) -> SpiceResult<Real> {
        let name = query.name();
        let device = self.point.device(self.circuit, query.device);
        let row_value =
            |row: Option<usize>| row.map_or(0., |row| self.request.solution.as_slice()[row]);
        let context = self.request.model_context;
        match &query.selector {
            Selector::Current { load, slot, signed } => {
                let current = self.current(*load, *slot, name)?;
                Ok(if *signed {
                    current * device.observation_current_sign()
                } else {
                    current
                })
            }
            Selector::Power { load, slot, rows } => {
                let current = self.current(*load, *slot, name)? * device.observation_current_sign();
                Ok(current * (row_value(rows[0]) - row_value(rows[1])))
            }
            Selector::SourceDc => self
                .sources
                .iter()
                .find(|(n, _)| n.eq_ignore_ascii_case(device.name()))
                .map(|(_, value)| *value)
                .or(device.observation_parameter("dc", context)?)
                .ok_or_else(|| gap(name)),
            Selector::Parameter(key) => {
                if let Some(value) = device.observation_parameter(key, context)? {
                    Ok(value)
                } else {
                    let voltage = |node| row_value(self.circuit.unknowns().node_row(node));
                    device
                        .observation_operating(key, context, &voltage, self.time.is_some())?
                        .ok_or_else(|| gap(name))
                }
            }
        }
    }
}

impl ObservationPlan {
    fn bind(circuit: &Circuit, names: &[String]) -> SpiceResult<Self> {
        let mut plan = Self {
            topology: Some(circuit.topology_id()?),
            ..Self::default()
        };
        let mut seen = BTreeSet::new();
        let mut load_by_device = BTreeMap::new();
        for name in names {
            let name = name.to_ascii_lowercase();
            let (instance, key) = query(&name)?;
            let index = circuit
                .observation_device_index(instance)
                .ok_or_else(|| SpiceError::circuit(format!("{name}: unknown device {instance}")))?;
            let device = circuit.devices()[index].as_ref();
            if name.starts_with("i(") && device.branch_currents() == 1 {
                continue;
            }
            if !seen.insert(name.clone()) {
                continue;
            }
            if plan.queries.len() >= 1024 {
                return Err(gap("more than 1024 requested quantities"));
            }
            let power = matches!(key, "p" | "power");
            let selector = if let Some(port) = terminal(device, key).or_else(|| power.then_some(0))
            {
                let terminals = device.terminals();
                if power && terminals.len() != 2 {
                    return Err(gap(&name));
                }
                let node = terminals.get(port).ok_or_else(|| gap(&name))?;
                if terminals
                    .iter()
                    .filter(|terminal| *terminal == node)
                    .count()
                    != 1
                {
                    return Err(gap(&format!("{name}: coincident terminals")));
                }
                let load = *load_by_device.entry(index).or_insert_with(|| {
                    let load = plan.loads.len();
                    plan.loads
                        .push(TerminalLayout::new(circuit, index, terminals));
                    load
                });
                let slot = plan.loads[load].ports[port];
                if power {
                    Selector::Power {
                        load,
                        slot,
                        rows: [
                            circuit.unknowns().node_row(terminals[0]),
                            circuit.unknowns().node_row(terminals[1]),
                        ],
                    }
                } else {
                    Selector::Current {
                        load,
                        slot,
                        signed: !matches!(key, "i" | "current"),
                    }
                }
            } else if key == "dc" && matches!(device.designator(), 'v' | 'i') {
                Selector::SourceDc
            } else {
                Selector::Parameter(key.into())
            };
            plan.queries.push(Observation {
                unit: unit(device.designator(), key),
                name,
                device: index,
                selector,
            });
        }
        Ok(plan)
    }

    fn validate(&self, circuit: &Circuit) -> SpiceResult<()> {
        let topology = circuit.topology_id()?;
        if self.topology.is_some_and(|bound| bound != topology) {
            return Err(SpiceError::circuit("stale observation topology binding"));
        }
        Ok(())
    }
}

impl Circuit {
    /// Bind extra quantities to finalized devices and terminal roles.
    /// Existing branch-current columns are retained without duplication.
    ///
    /// # Errors
    /// Unknown devices, invalid terminal queries, malformed queries or
    /// more than 1024 distinct additional quantities.
    pub fn set_observations(&mut self, names: &[String]) -> SpiceResult<()> {
        self.runtime.observations = ObservationPlan::bind(self, names)?;
        Ok(())
    }

    /// Extra bound columns requested through [`Self::set_observations`].
    #[must_use]
    pub fn observations(&self) -> &[Observation] {
        &self.runtime.observations.queries
    }

    /// Observe a converged real point using isolated, disposable device loads.
    /// The pre-acceptance history and coefficients must be those of the solved trial.
    ///
    /// # Errors
    /// Unsupported quantities, stale state, or nonfinite arithmetic.
    pub fn observe_real(
        &self,
        request: &LoadRequest<'_>,
        sources: &[(&str, Real)],
        time: Option<(Real, Forcing)>,
        previous: Option<&TrialState>,
    ) -> SpiceResult<Vec<Complex>> {
        self.prepare_run()
            .observe_real(request, sources, time, previous)
    }

    fn observe_real_prepared(
        &self,
        run: &PreparedRun<'_>,
        request: &LoadRequest<'_>,
        sources: &[(&str, Real)],
        time: Option<(Real, Forcing)>,
        previous: Option<&TrialState>,
    ) -> SpiceResult<Vec<Complex>> {
        self.validate_history(request.history)?;
        if let Some(trial) = previous {
            request.history.check_trial(trial)?;
        }
        let plan = &self.runtime.observations;
        plan.validate(self)?;
        if plan.queries.is_empty() {
            return Ok(Vec::new());
        }
        if request.solution.len() != self.unknown_count() {
            return Err(SpiceError::circuit(
                "isolated observation load does not match the circuit numbering",
            ));
        }
        let point = run.point(request.model_context)?;
        let mut workspace = run.observation_work.borrow_mut();
        let workspace = workspace.get_or_insert_with(|| ObservationWorkspace::new(self, plan));
        workspace.begin();
        let mut loads = PointLoads {
            circuit: self,
            request,
            sources,
            time,
            previous,
            point: &point,
            mutual: None,
            workspace,
        };
        plan.queries
            .iter()
            .map(|query| {
                let value = loads.observe(query)?;
                if !value.is_finite() {
                    return Err(SpiceError::Numerical {
                        context: query.name.clone(),
                        message: "nonfinite device observation".into(),
                    });
                }
                Ok(Complex::real(value))
            })
            .collect()
    }

    /// Observe scalar instance parameters for a spectral point. Device current
    /// and power asks are rejected as by C's passive asks in AC.
    ///
    /// # Errors
    /// A current/power or unsupported scalar query, stale plan, or nonfinite parameter.
    pub fn observe_parameters(&self, context: &ModelContext) -> SpiceResult<Vec<Complex>> {
        self.prepare_run().observe_parameters(context)
    }

    fn observe_parameters_prepared(
        &self,
        run: &PreparedRun<'_>,
        context: &ModelContext,
    ) -> SpiceResult<Vec<Complex>> {
        let plan = &self.runtime.observations;
        plan.validate(self)?;
        let point = run.point(context)?;
        plan.queries
            .iter()
            .map(|query| {
                let key = match &query.selector {
                    Selector::SourceDc => "dc",
                    Selector::Parameter(key) => key,
                    Selector::Current { .. } | Selector::Power { .. } => {
                        return Err(gap(&format!("{}: AC current/power ask", query.name())));
                    }
                };
                let value = point
                    .device(self, query.device)
                    .observation_parameter(key, context)?
                    .ok_or_else(|| gap(query.name()))?;
                if !value.is_finite() {
                    return Err(gap(query.name()));
                }
                Ok(Complex::real(value))
            })
            .collect()
    }
}

impl PreparedRun<'_> {
    /// Observe a real point with the run's already prepared devices.
    /// # Errors
    /// As [`Circuit::observe_real`].
    pub fn observe_real(
        &self,
        request: &LoadRequest<'_>,
        sources: &[(&str, Real)],
        time: Option<(Real, Forcing)>,
        previous: Option<&TrialState>,
    ) -> SpiceResult<Vec<Complex>> {
        self.circuit()
            .observe_real_prepared(self, request, sources, time, previous)
    }

    /// Observe scalar device parameters in the prepared context.
    /// # Errors
    /// As [`Circuit::observe_parameters`].
    pub fn observe_parameters(&self, context: &ModelContext) -> SpiceResult<Vec<Complex>> {
        self.circuit().observe_parameters_prepared(self, context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::devices::{AnalysisMode, StampContext};
    use std::{cell::Cell, rc::Rc};

    #[derive(Debug)]
    struct Counted {
        name: &'static str,
        nodes: [NodeId; 2],
        count: Rc<Cell<usize>>,
    }

    impl Device for Counted {
        fn name(&self) -> &str {
            self.name
        }
        fn designator(&self) -> char {
            'r'
        }
        fn terminals(&self) -> &[NodeId] {
            &self.nodes
        }
        fn stamp(&self, context: &mut StampContext<'_>) -> SpiceResult<()> {
            self.count.set(self.count.get() + 1);
            let [a, b] = self.nodes;
            for (row, col, value) in [(a, a, 2.), (a, b, -2.), (b, a, -2.), (b, b, 2.)] {
                context.stamp(row, col, value)?;
            }
            Ok(())
        }
    }

    #[test]
    fn each_observed_device_loads_once_per_point_even_with_interleaved_queries() {
        let mut circuit = Circuit::new();
        let a = circuit.add_node("a").unwrap();
        let b = circuit.add_node("b").unwrap();
        let first = Rc::new(Cell::new(0));
        let second = Rc::new(Cell::new(0));
        for (name, nodes, count) in [
            ("r1", [a, b], Rc::clone(&first)),
            ("r2", [NodeId::GROUND, a], Rc::clone(&second)),
        ] {
            circuit
                .add_device(Box::new(Counted { name, nodes, count }))
                .unwrap();
        }
        circuit.finalize().unwrap();
        circuit
            .set_observations(&[
                "i(r1)".into(),
                "@r2[i]".into(),
                "@r1[p]".into(),
                "@r2[power]".into(),
                "@r1[current]".into(),
                "I(R1)".into(),
            ])
            .unwrap();
        assert_eq!(circuit.observations().len(), 5);
        let history = circuit.state_history().unwrap();
        let model = ModelContext::default();
        let names: Vec<_> = circuit
            .observations()
            .iter()
            .map(|q| (q.name(), q.unit()))
            .collect();
        assert_eq!(
            names,
            [
                ("i(r1)", "current"),
                ("@r2[i]", "current"),
                ("@r1[p]", "power"),
                ("@r2[power]", "power"),
                ("@r1[current]", "current"),
            ]
        );
        let counts = [first.get(), second.get()];
        let run = circuit.prepare_run();
        for (step, (solution, expected)) in [
            ([3., 1.], [4., -6., 8., 18., 4.]),
            ([-1., 2.], [-6., 2., 18., 2., -6.]),
            ([0., 0.], [0.; 5]),
        ]
        .into_iter()
        .enumerate()
        {
            let solution = Vector::from_slice(&solution);
            let request = LoadRequest {
                mode: AnalysisMode::OperatingPoint,
                solution: &solution,
                model_context: &model,
                integration: None,
                history: &history,
                forcing: None,
            };
            let values = run.observe_real(&request, &[], None, None).unwrap();
            assert_eq!(values, expected.map(Complex::real));
            assert_eq!(
                [first.get(), second.get()],
                [counts[0] + step + 1, counts[1] + step + 1]
            );
        }
        assert!(
            run.observe_parameters(&model)
                .unwrap_err()
                .to_string()
                .contains("AC current/power")
        );
    }

    #[derive(Debug)]
    struct Stateful {
        name: &'static str,
        nodes: [NodeId; 2],
        slots: usize,
    }

    impl Device for Stateful {
        fn name(&self) -> &str {
            self.name
        }
        fn designator(&self) -> char {
            'r'
        }
        fn terminals(&self) -> &[NodeId] {
            &self.nodes
        }
        fn state_count(&self) -> usize {
            self.slots
        }
        fn stamp(&self, context: &mut StampContext<'_>) -> SpiceResult<()> {
            assert_eq!(context.states.len(), self.slots);
            assert!(context.states.trial(0).unwrap().is_nan());
            assert!(!context.states.device_limiting());
            let value = context.states.accepted(1, 0).unwrap_or(0.)
                + context.states.iterate(0).unwrap_or(0.);
            context.states.set(0, value)?;
            context.states.patch_state1(0, value + 100.)?;
            context.states.report_nonconvergence()?;
            context.stamp_rhs(self.nodes[0], -value)?;
            context.stamp_rhs(self.nodes[1], value)
        }
    }

    #[test]
    fn isolated_state_and_rhs_are_point_local_even_with_previous_patches() {
        let mut circuit = Circuit::new();
        let a = circuit.add_node("a").unwrap();
        let b = circuit.add_node("b").unwrap();
        for (name, node, slots) in [("r1", a, 1), ("r2", b, 4)] {
            circuit
                .add_device(Box::new(Stateful {
                    name,
                    nodes: [node, NodeId::GROUND],
                    slots,
                }))
                .unwrap();
        }
        circuit.finalize().unwrap();
        circuit
            .set_observations(&["i(r1)".into(), "i(r2)".into(), "@r1[p]".into()])
            .unwrap();
        let mut history = circuit.state_history().unwrap();
        let mut accepted = history.trial();
        for index in 0..2 {
            let mut view = history.device_of(&mut accepted, index).unwrap();
            for slot in 0..view.len() {
                view.set(slot, if index == 0 { 2. } else { 3. }).unwrap();
            }
        }
        history.commit(accepted).unwrap();
        let mut previous = history.trial();
        for (index, value) in [5., 7.].into_iter().enumerate() {
            let mut view = history.device_of(&mut previous, index).unwrap();
            view.set(0, value).unwrap();
            view.patch_state1(0, value + 10.).unwrap();
        }
        let history_before = history.clone();
        let previous_before = previous.clone();
        let run = circuit.prepare_run();
        let model = ModelContext::default();
        let solution = Vector::from_slice(&[2., 3.]);
        let request = LoadRequest {
            mode: AnalysisMode::OperatingPoint,
            solution: &solution,
            model_context: &model,
            integration: None,
            history: &history,
            forcing: None,
        };
        for (prev, expected) in [
            (Some(&previous), [20., 24., 40.]),
            (None, [2., 3., 4.]),
            (Some(&previous), [20., 24., 40.]),
        ] {
            assert_eq!(
                run.observe_real(&request, &[], None, prev).unwrap(),
                expected.map(Complex::real)
            );
        }
        assert_eq!(history, history_before);
        // NaN slots cannot be compared using TrialState's PartialEq.
        assert_eq!(
            previous
                .values()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            previous_before
                .values()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        let work = run.observation_work.borrow();
        let work = work.as_ref().unwrap();
        assert!(work.loaded.iter().all(|load| load.ready));
    }

    #[test]
    fn binding_failures_are_transactional_and_topology_edits_invalidate() {
        let mut circuit = Circuit::new();
        let a = circuit.add_node("a").unwrap();
        let g = NodeId::GROUND;
        let count = Rc::new(Cell::new(0));
        for (name, nodes) in [("r1", [a, g]), ("r2", [a, a])] {
            circuit
                .add_device(Box::new(Counted {
                    name,
                    nodes,
                    count: Rc::clone(&count),
                }))
                .unwrap();
        }
        circuit.finalize().unwrap();
        circuit.set_observations(&["i(r1)".into()]).unwrap();
        for query in ["i(unknown)", "@r1[", "i(r2)", "@r2[p]", "@r1[ib]]"] {
            assert!(
                circuit.set_observations(&[query.into()]).is_err(),
                "{query}"
            );
            assert_eq!(circuit.observations()[0].name(), "i(r1)");
        }
        let history = circuit.state_history().unwrap();
        let model = ModelContext::default();
        let solution = Vector::from_slice(&[1.]);
        let request = LoadRequest {
            mode: AnalysisMode::OperatingPoint,
            solution: &solution,
            model_context: &model,
            integration: None,
            history: &history,
            forcing: None,
        };
        circuit
            .replace_device(
                0,
                Box::new(Counted {
                    name: "r1",
                    nodes: [g, a],
                    count,
                }),
            )
            .unwrap();
        assert!(circuit.observe_real(&request, &[], None, None).is_err());
        circuit.finalize().unwrap();
        assert!(circuit.observations().is_empty());
        assert!(circuit.observe_real(&request, &[], None, None).is_err());
        circuit.set_observations(&["i(r1)".into()]).unwrap();
        let history = circuit.state_history().unwrap();
        let request = LoadRequest {
            history: &history,
            ..request
        };
        assert_eq!(
            circuit.observe_real(&request, &[], None, None).unwrap(),
            [Complex::real(-2.)]
        );
    }

    #[test]
    fn a_nonfinite_point_does_not_poison_the_next_observation() {
        let mut circuit = Circuit::new();
        let a = circuit.add_node("a").unwrap();
        circuit
            .add_device(Box::new(Counted {
                name: "r1",
                nodes: [a, NodeId::GROUND],
                count: Rc::new(Cell::new(0)),
            }))
            .unwrap();
        circuit.finalize().unwrap();
        circuit.set_observations(&["i(r1)".into()]).unwrap();
        let history = circuit.state_history().unwrap();
        let model = ModelContext::default();
        let run = circuit.prepare_run();
        for (value, expected) in [(Real::MAX, None), (2., Some(4.))] {
            let solution = Vector::from_slice(&[value]);
            let request = LoadRequest {
                mode: AnalysisMode::OperatingPoint,
                solution: &solution,
                model_context: &model,
                integration: None,
                history: &history,
                forcing: None,
            };
            let actual = run.observe_real(&request, &[], None, None);
            if let Some(expected) = expected {
                assert_eq!(actual.unwrap(), [Complex::real(expected)]);
            } else {
                assert!(matches!(actual, Err(SpiceError::Numerical { .. })));
            }
        }
    }

    #[test]
    fn terminal_residuals_match_full_multiply_with_aliases_and_cancellation() {
        let mut circuit = Circuit::new();
        let a = circuit.add_node("a").unwrap();
        let b = circuit.add_node("b").unwrap();
        let g = NodeId::GROUND;
        circuit.finalize().unwrap();
        let mut matrix = SparseMatrix::new(2, 2);
        for (row, col, value) in [
            (0, 1, 1e16),
            (1, 0, -3.),
            (0, 0, 2.),
            (0, 1, -1e16),
            (0, 1, 1.),
        ] {
            matrix.add(row, col, value).unwrap();
        }
        let solution = Vector::from_slice(&[2., 1.]);
        let rhs = Vector::from_slice(&[0.25, -1.]);
        let ax = matrix.mul_vector(&solution).unwrap();
        let ra = ax.as_slice()[0] - rhs.as_slice()[0];
        let rb = ax.as_slice()[1] - rhs.as_slice()[1];
        let layout = TerminalLayout::new(&circuit, 0, &[g, a, b, a]);
        let mut currents = vec![0.; layout.rows.len() + 1];
        layout.residuals(&matrix, |row| rhs.as_slice()[row], &solution, &mut currents);
        let actual = layout.ports.iter().map(|&slot| currents[slot]);
        for (got, expected) in actual.zip([0. - ra - rb, ra, rb, ra]) {
            assert_eq!(got.to_bits(), expected.to_bits());
        }
    }
}
