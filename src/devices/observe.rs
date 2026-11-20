//! Device observations from the same physical loads used by the solver.
//!
//! C: `frontend/outitf.c`, `spiceif.c::if_getparam`, device `*ask.c` routines.
//! Loads are disposable; no observation accepts state or advances history.
use std::collections::{BTreeMap, BTreeSet};

use crate::devices::{
    Circuit, Device, Forcing, IterationPhase, LoadRequest, ModelContext, TrialState,
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

/// Rawfile units of the supported observation quantities.
#[must_use]
pub fn unit(name: &str) -> &'static str {
    let (device, key) = query(name).unwrap_or(("", ""));
    if key == "ic" && device.starts_with('c') {
        return "voltage";
    }
    // mesask.c: drain, gate and gate-drain currents.
    if matches!(key, "cd" | "cg" | "cgd") && device.starts_with('z') {
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

/// Resolve observation targets once, while keeping the public column names
/// in their original order. A caller replacing devices through `devices_mut`
/// can invalidate a hint; identity is checked before using it.
#[derive(Default)]
pub(super) struct ObservationPlan {
    names: Vec<String>,
    devices: Vec<usize>,
}

impl ObservationPlan {
    fn iter(&self) -> impl Iterator<Item = (&str, usize)> {
        self.names
            .iter()
            .map(String::as_str)
            .zip(self.devices.iter().copied())
    }
}

fn observation_index(circuit: &Circuit, instance: &str, hint: usize) -> Option<usize> {
    if circuit
        .devices()
        .get(hint)
        .is_some_and(|d| d.name().eq_ignore_ascii_case(instance))
    {
        Some(hint)
    } else {
        circuit
            .devices()
            .iter()
            .position(|d| d.name().eq_ignore_ascii_case(instance))
    }
}

/// One converged point: the isolated matrix and RHS are scratch, not accepted
/// state. Cache only the terminal residuals, not a dense vector per device.
struct PointLoads<'a, 'b> {
    circuit: &'a Circuit,
    request: &'a LoadRequest<'b>,
    sources: &'a [(&'a str, Real)],
    time: Option<(Real, Forcing)>,
    previous: Option<&'a TrialState>,
    replacements: &'a [(usize, Box<dyn Device>)],
    mutual: Option<crate::devices::circuit::PreparedMutualTerms>,
    scratch: Option<(SparseMatrix, Vector)>,
    currents: BTreeMap<usize, Vec<Real>>,
}

impl PointLoads<'_, '_> {
    fn current(
        &mut self,
        index: usize,
        device: &dyn Device,
        port: usize,
        query: &str,
    ) -> SpiceResult<Real> {
        if let Some(currents) = self.currents.get(&index) {
            return currents.get(port).copied().ok_or_else(|| gap(query));
        }
        if self.mutual.is_none() {
            self.circuit.check_numbering()?;
            let n = self.circuit.unknown_count();
            if self.request.solution.len() != n
                || self.request.history.len() != self.circuit.state_len()
            {
                return Err(SpiceError::circuit(
                    "isolated observation load does not match the circuit numbering",
                ));
            }
            self.mutual =
                Some(self.circuit.mutual_terms_with_replacements(
                    self.request.model_context,
                    self.replacements,
                )?);
        }
        let n = self.circuit.unknown_count();
        let (matrix, rhs) = self
            .scratch
            .get_or_insert_with(|| (SparseMatrix::new(n, n), Vector::zeros(n)));
        matrix.clear();
        rhs.clear();
        let mut trial = self
            .request
            .history
            .trial_in(IterationPhase::Float, self.previous)?
            .with_device_limiting(false);
        self.circuit.stamp_isolated(
            index,
            device,
            self.request,
            matrix,
            rhs,
            &mut trial,
            self.mutual.as_deref().unwrap_or(&[]),
        )?;
        if device.designator() == 'i' {
            let value = device
                .observation_source_current(self.time)?
                .ok_or_else(|| gap(query))?;
            let value = self
                .sources
                .iter()
                .find(|(name, _)| name.eq_ignore_ascii_case(device.name()))
                .map_or(value, |(_, v)| *v * device.observation_source_multiplier());
            for (terminal, sign) in device.terminals().iter().take(2).zip([-1., 1.]) {
                if let Some(row) = self.circuit.unknowns().node_row(*terminal) {
                    rhs.as_mut_slice()[row] = sign * value;
                }
            }
        }
        let currents = terminal_residuals(
            self.circuit,
            device.terminals(),
            matrix,
            rhs,
            self.request.solution,
        );
        let value = currents.get(port).copied().ok_or_else(|| gap(query))?;
        self.currents.insert(index, currents);
        Ok(value)
    }
}

/// Same accumulation as `SparseMatrix::mul_vector`, but only for terminal
/// rows. Preserve the unfolded contribution order, including cancellation
/// and aliased nodes. Ground is the negative sum in terminal order, with
/// each physical node counted once.
fn terminal_residuals(
    circuit: &Circuit,
    terminals: &[NodeId],
    matrix: &SparseMatrix,
    rhs: &Vector,
    solution: &Vector,
) -> Vec<Real> {
    let rows: Vec<_> = terminals
        .iter()
        .map(|&node| circuit.unknowns().node_row(node))
        .collect();
    let mut currents = vec![0.; rows.len()];
    for triplet in matrix.triplets() {
        for (&row, current) in rows.iter().zip(&mut currents) {
            if row == Some(triplet.row) {
                let contribution = triplet.value * solution.as_slice()[triplet.col];
                *current += contribution;
            }
        }
    }
    for (&row, current) in rows.iter().zip(&mut currents) {
        if let Some(row) = row {
            *current -= rhs.as_slice()[row];
        }
    }
    let mut ground = 0.;
    for (port, node) in terminals.iter().enumerate() {
        if rows[port].is_some() && !terminals[..port].contains(node) {
            ground -= currents[port];
        }
    }
    for (&row, current) in rows.iter().zip(&mut currents) {
        if row.is_none() {
            *current = ground;
        }
    }
    currents
}

impl Circuit {
    /// Select extra device quantities to record at each solved point.
    /// Existing branch-current columns are retained without duplication.
    ///
    /// # Errors
    /// Unknown devices, malformed queries, or duplicate queries beyond the budget.
    pub fn set_observations(&mut self, names: &[String]) -> SpiceResult<()> {
        let mut plan = ObservationPlan::default();
        let mut seen = BTreeSet::new();
        for name in names {
            let name = name.to_ascii_lowercase();
            let (device, _) = query(&name)?;
            let index = self
                .devices()
                .iter()
                .position(|d| d.name().eq_ignore_ascii_case(device))
                .ok_or_else(|| SpiceError::circuit(format!("{name}: unknown device {device}")))?;
            if name.starts_with("i(") && self.devices()[index].branch_currents() == 1 {
                continue;
            }
            if seen.insert(name.clone()) {
                plan.names.push(name);
                plan.devices.push(index);
            }
            if plan.names.len() > 1024 {
                return Err(gap("more than 1024 requested quantities"));
            }
        }
        self.observations = plan;
        Ok(())
    }

    /// Extra columns requested through [`Self::set_observations`].
    #[must_use]
    pub fn observations(&self) -> &[String] {
        &self.observations.names
    }

    /// Observe a converged real point using immutable copies of its device loads.
    /// `sources` carries DC sweep source values; `time` selects transient source forcing.
    /// The pre-acceptance history and coefficients must be those of the solved trial.
    ///
    /// # Errors
    /// Unsupported quantities, ambiguous coincident terminal currents, or nonfinite arithmetic.
    pub fn observe_real(
        &self,
        request: &LoadRequest<'_>,
        sources: &[(&str, Real)],
        time: Option<(Real, crate::devices::Forcing)>,
        previous: Option<&crate::devices::TrialState>,
    ) -> SpiceResult<Vec<Complex>> {
        if self.observations.names.is_empty() {
            return Ok(Vec::new());
        }
        let replacements = self.replacements(request.model_context)?;
        let mut loads = PointLoads {
            circuit: self,
            request,
            sources,
            time,
            previous,
            replacements: &replacements,
            mutual: None,
            scratch: None,
            currents: BTreeMap::new(),
        };
        self.observations
            .iter()
            .map(|(name, hint)| {
                let (instance, key) = query(name)?;
                let index = observation_index(self, instance, hint).ok_or_else(|| {
                    SpiceError::circuit(format!("unknown observation device {instance}"))
                })?;
                let device: &dyn Device = replacements
                    .iter()
                    .find(|(i, _)| *i == index)
                    .map_or_else(|| self.devices()[index].as_ref(), |(_, d)| d.as_ref());
                let power = matches!(key, "p" | "power");
                let value = if let Some(port) = terminal(device, key).or_else(|| power.then_some(0))
                {
                    if power && device.terminals().len() != 2 {
                        return Err(gap(name));
                    }
                    let node = *device.terminals().get(port).ok_or_else(|| gap(name))?;
                    if device.terminals().iter().filter(|n| **n == node).count() != 1 {
                        return Err(gap(&format!("{name}: coincident terminals")));
                    }
                    let current = loads.current(index, device, port, name)?;
                    let current = if matches!(key, "i" | "current") {
                        current
                    } else {
                        current * device.observation_current_sign()
                    };
                    if power {
                        let voltage = |node| {
                            self.unknowns()
                                .node_row(node)
                                .map_or(0., |r| request.solution.as_slice()[r])
                        };
                        current * (voltage(device.terminals()[0]) - voltage(device.terminals()[1]))
                    } else {
                        current
                    }
                } else if key == "dc" && matches!(device.designator(), 'v' | 'i') {
                    sources
                        .iter()
                        .find(|(n, _)| n.eq_ignore_ascii_case(instance))
                        .map(|(_, value)| *value)
                        .or(device.observation_parameter(key, request.model_context)?)
                        .ok_or_else(|| gap(name))?
                } else if let Some(value) =
                    device.observation_parameter(key, request.model_context)?
                {
                    value
                } else {
                    let voltage = |node: crate::primitives::NodeId| {
                        self.unknowns()
                            .node_row(node)
                            .map_or(0., |r| request.solution.as_slice()[r])
                    };
                    device
                        .observation_operating(
                            key,
                            request.model_context,
                            &voltage,
                            time.is_some(),
                        )?
                        .ok_or_else(|| gap(name))?
                };
                if !value.is_finite() {
                    return Err(SpiceError::Numerical {
                        context: name.to_owned(),
                        message: "nonfinite device observation".into(),
                    });
                }
                Ok(Complex::real(value))
            })
            .collect()
    }

    /// Observe scalar instance parameters for a spectral point. Device current and
    /// power asks are rejected as by C's passive `*ask.c` routines in AC.
    ///
    /// # Errors
    /// A current/power or unsupported scalar query, or nonfinite parameter.
    pub fn observe_parameters(&self, context: &ModelContext) -> SpiceResult<Vec<Complex>> {
        let replacements = self.replacements(context)?;
        self.observations
            .iter()
            .map(|(name, hint)| {
                let (instance, key) = query(name)?;
                let index = observation_index(self, instance, hint).ok_or_else(|| gap(name))?;
                let device: &dyn Device = replacements
                    .iter()
                    .find(|(i, _)| *i == index)
                    .map_or_else(|| self.devices()[index].as_ref(), |(_, d)| d.as_ref());
                if terminal(device, key).is_some() || matches!(key, "p" | "power") {
                    return Err(gap(&format!("{name}: AC current/power ask")));
                }
                let value = device
                    .observation_parameter(key, context)?
                    .ok_or_else(|| gap(name))?;
                if !value.is_finite() {
                    return Err(gap(name));
                }
                Ok(Complex::real(value))
            })
            .collect()
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
        let a = circuit.add_node("a");
        let b = circuit.add_node("b");
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
        let history = circuit.state_history();
        let model = ModelContext::default();
        let solution = Vector::from_slice(&[3., 1.]);
        let request = LoadRequest {
            mode: AnalysisMode::OperatingPoint,
            solution: &solution,
            model_context: &model,
            integration: None,
            history: &history,
            forcing: None,
        };
        let counts = [first.get(), second.get()];
        for step in 1..=2 {
            let values = circuit.observe_real(&request, &[], None, None).unwrap();
            assert_eq!(values, [4., -6., 8., 18., 4.].map(Complex::real));
            assert_eq!(
                [first.get(), second.get()],
                [counts[0] + step, counts[1] + step]
            );
        }
    }

    #[test]
    fn terminal_residuals_match_full_multiply_with_aliases_and_cancellation() {
        let mut circuit = Circuit::new();
        let a = circuit.add_node("a");
        let b = circuit.add_node("b");
        let g = NodeId::GROUND;
        circuit.rebuild_unknowns();
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
        let actual = terminal_residuals(&circuit, &[g, a, b, a], &matrix, &rhs, &solution);
        for (got, expected) in actual.iter().zip([0. - ra - rb, ra, rb, ra]) {
            assert_eq!(got.to_bits(), expected.to_bits());
        }
    }
}
