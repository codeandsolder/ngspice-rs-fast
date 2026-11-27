//! Physical two-terminal storage networks, bound to ordinary native RLC
//! devices. Source-dependent defaults and current observation conventions are
//! resolved separately; the winding, storage and terminal identities are
//! intentionally distinct.

use crate::devices::{
    Capacitor, Circuit, Device, IndependentSource, Inductor, Resistor, ResistorNoise, StorageKind,
    Waveform,
    elaboration::{ElaborationSize, reserve},
    passive_bias::PassiveInitialization,
    rlc::REACTIVE_STATE_COUNT,
};
use crate::primitives::{Complex, NodeAppend, NodeId, NodeKind, Real, SpiceError, SpiceResult};

/// Resolved physical coefficients of one parallel copy of a storage network.
///
/// The main storage is in series with the series resistance and auxiliary
/// inductance. Shunts may span the auxiliary inductance, the main storage or
/// the entire terminal pair. A zero series coefficient collapses its edge
/// exactly; an absent shunt is an open circuit, not a large resistance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PassiveNetworkCoefficients {
    /// Main storage family.
    pub kind: StorageKind,
    /// Main capacitance or inductance.
    pub value: Real,
    /// Series resistance in ohms (zero means no series resistor).
    pub series_resistance: Real,
    /// Additional series inductance in henries.
    pub series_inductance: Real,
    /// Resistance across the additional series inductance, if present.
    pub series_inductance_shunt: Option<Real>,
    /// Resistance directly across the main storage, if present.
    pub storage_shunt: Option<Real>,
    /// Resistance across the external terminal pair, if present.
    pub parallel_resistance: Option<Real>,
    /// Capacitance across the external terminal pair.
    pub parallel_capacitance: Real,
    /// Number of identical parallel copies; zero removes the network.
    pub multiplicity: Real,
    /// Resolved resistor noise temperature in Celsius, or circuit temperature.
    pub temperature: Option<Real>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    SeriesResistance,
    SeriesInductance,
    SeriesInductanceShunt,
    Primary,
    StorageShunt,
    ParallelResistance,
    ParallelCapacitance,
}

impl Role {
    fn name(self, primary: &str) -> String {
        match self {
            Self::SeriesResistance => format!("r{primary}#series"),
            Self::SeriesInductance => format!("l{primary}#series"),
            Self::SeriesInductanceShunt => format!("r{primary}#series_shunt"),
            Self::Primary => primary.to_owned(),
            Self::StorageShunt => format!("r{primary}#storage_shunt"),
            Self::ParallelResistance => format!("r{primary}#parallel"),
            Self::ParallelCapacitance => format!("c{primary}#parallel"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Element {
    Resistance(Real),
    Storage(StorageKind, Real),
}

impl Element {
    const fn state_slots(self) -> usize {
        match self {
            Self::Resistance(_) => 0,
            Self::Storage(..) => REACTIVE_STATE_COUNT,
        }
    }

    fn scaled(self, multiplicity: Real) -> SpiceResult<Self> {
        let (original, scaled) = match self {
            Self::Resistance(r) => (r, Self::Resistance(r / multiplicity)),
            Self::Storage(kind, v) => (
                v,
                Self::Storage(
                    kind,
                    match kind {
                        StorageKind::Capacitor => v * multiplicity,
                        StorageKind::Inductor => v / multiplicity,
                    },
                ),
            ),
        };
        let value = match scaled {
            Self::Resistance(r) | Self::Storage(_, r) => r,
        };
        if !original.is_finite()
            || original < 0.
            || !value.is_finite()
            || (original != 0. && value == 0.)
            || matches!(scaled, Self::Resistance(r) if r == 0. || !(1. / r).is_finite())
        {
            return Err(SpiceError::circuit(
                "invalid scaled passive network coefficient",
            ));
        }
        Ok(scaled)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Edge {
    role: Role,
    element: Element,
    // Local nodes 0 and 1 are the sensed positive and external negative
    // terminals, respectively. All other nodes are physical series junctions.
    nodes: [usize; 2],
}

/// Validated, occurrence-specific network. No source parsing or graph
/// construction is required during analysis or device loading.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedPassiveNetwork {
    edges: Vec<Edge>,
    nodes: usize,
    temperature: Option<Real>,
    initialization: Option<PassiveInitialization>,
}

/// Stable names for the independently observable currents of a bound network.
///
/// The primary device keeps the original name, including for K coupling to
/// an inductor. The terminal probe measures the sum of all parallel paths.
/// Source-specific observation policy must select the appropriate identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PassiveNetworkBinding {
    /// Main charge/flux device identity.
    pub primary: String,
    /// Zero-voltage port probe, oriented into the positive external terminal.
    pub terminal_probe: String,
}

impl PreparedPassiveNetwork {
    /// Resolve multiplicity, remove zero series elements and build the
    /// immutable native topology. The optional initialization is already in
    /// aggregate physical coordinates (e.g. total winding current).
    ///
    /// # Errors
    /// Invalid physical coefficients, temperature, multiplicity, scaled
    /// values, or a startup constraint incompatible with the storage family.
    pub fn prepare(
        input: PassiveNetworkCoefficients,
        initialization: Option<PassiveInitialization>,
    ) -> SpiceResult<Option<Self>> {
        if !input.multiplicity.is_finite()
            || input.multiplicity < 0.
            || input
                .temperature
                .is_some_and(|t| !t.is_finite() || t <= -273.15)
            || [
                input.value,
                input.series_resistance,
                input.series_inductance,
                input.parallel_capacitance,
            ]
            .iter()
            .any(|v| !v.is_finite() || *v < 0.)
            || [
                input.series_inductance_shunt,
                input.storage_shunt,
                input.parallel_resistance,
            ]
            .into_iter()
            .flatten()
            .any(|r| !r.is_finite() || r <= 0. || !(1. / r).is_finite())
        {
            return Err(SpiceError::circuit("invalid passive network coefficients"));
        }
        if let Some(policy) = &initialization {
            policy.check_kind(input.kind)?;
        }
        if input.multiplicity == 0. {
            return Ok(None);
        }

        let mut edges = Vec::new();
        reserve(&mut edges, 7, None)?;
        let mut append = |role, element: Element, nodes| -> SpiceResult<()> {
            edges.push(Edge {
                role,
                element: element.scaled(input.multiplicity)?,
                nodes,
            });
            Ok(())
        };
        let mut nodes = 2;
        let mut positive = 0;
        if input.series_resistance != 0. {
            append(
                Role::SeriesResistance,
                Element::Resistance(input.series_resistance),
                [positive, nodes],
            )?;
            positive = nodes;
            nodes += 1;
        }
        if input.series_inductance != 0. {
            append(
                Role::SeriesInductance,
                Element::Storage(StorageKind::Inductor, input.series_inductance),
                [positive, nodes],
            )?;
            if let Some(r) = input.series_inductance_shunt {
                append(
                    Role::SeriesInductanceShunt,
                    Element::Resistance(r),
                    [positive, nodes],
                )?;
            }
            positive = nodes;
            nodes += 1;
        }
        append(
            Role::Primary,
            Element::Storage(input.kind, input.value),
            [positive, 1],
        )?;
        if let Some(r) = input.storage_shunt {
            append(Role::StorageShunt, Element::Resistance(r), [positive, 1])?;
        }
        if let Some(r) = input.parallel_resistance {
            append(Role::ParallelResistance, Element::Resistance(r), [0, 1])?;
        }
        if input.parallel_capacitance != 0. {
            append(
                Role::ParallelCapacitance,
                Element::Storage(StorageKind::Capacitor, input.parallel_capacitance),
                [0, 1],
            )?;
        }
        Ok(Some(Self {
            edges,
            nodes,
            temperature: input.temperature,
            initialization,
        }))
    }

    /// Atomically materialize the validated network and its separate current
    /// identities using the simulator's native charge, flux and noise devices.
    /// The zero-voltage probe introduces an alias node, not an additional
    /// physical shunt site.
    ///
    /// # Errors
    /// Invalid terminals, duplicate identities or exhausted elaboration limits.
    pub fn bind(
        &self,
        circuit: &mut Circuit,
        name: &str,
        terminals: [NodeId; 2],
    ) -> SpiceResult<PassiveNetworkBinding> {
        if name.is_empty()
            || terminals
                .iter()
                .any(|&id| circuit.nodes().node(id).is_none())
        {
            return Err(SpiceError::circuit("invalid passive network binding"));
        }
        let binding = PassiveNetworkBinding {
            primary: name.to_owned(),
            terminal_probe: format!("v{name}#port"),
        };
        let size = ElaborationSize::new(
            self.edges.len() + 1,
            self.nodes - 1,
            self.edges
                .iter()
                .map(|edge| edge.element.state_slots())
                .sum(),
        );
        circuit.add_prepared_devices(size, |nodes| self.build(nodes, &binding, terminals))?;
        Ok(binding)
    }

    fn build(
        &self,
        nodes: &mut NodeAppend<'_>,
        binding: &PassiveNetworkBinding,
        terminals: [NodeId; 2],
    ) -> SpiceResult<Vec<Box<dyn Device>>> {
        let [positive, negative] = terminals;
        let mut ids = Vec::new();
        reserve(&mut ids, self.nodes, None)?;
        let alias = new_node(nodes, &binding.primary, 0, NodeKind::ProbeAlias)?;
        ids.extend([alias, negative]);
        for index in 2..self.nodes {
            ids.push(new_node(
                nodes,
                &binding.primary,
                index,
                NodeKind::Internal,
            )?);
        }
        let mut devices: Vec<Box<dyn Device>> = Vec::new();
        reserve(&mut devices, self.edges.len() + 1, None)?;
        devices.push(Box::new(IndependentSource::new(
            &binding.terminal_probe,
            [positive, alias],
            true,
            0.,
            Complex::ZERO,
            Waveform::Constant(0.),
        )?));
        for edge in &self.edges {
            let name = edge.role.name(&binding.primary);
            let terminals = edge.nodes.map(|index| ids[index]);
            let mut device: Box<dyn Device> = match edge.element {
                Element::Resistance(r) => Box::new(Resistor::new(name, terminals, r)?.with_noise(
                    ResistorNoise {
                        temperature: self.temperature,
                        ..Default::default()
                    },
                )),
                Element::Storage(StorageKind::Capacitor, c) => {
                    Box::new(Capacitor::new(name, terminals, c, None)?)
                }
                Element::Storage(StorageKind::Inductor, l) => {
                    Box::new(Inductor::new(name, terminals, l, None)?)
                }
            };
            if edge.role == Role::Primary
                && let Some(policy) = &self.initialization
            {
                device.bind_passive_initialization(policy.clone())?;
            }
            devices.push(device);
        }
        Ok(devices)
    }
}

fn new_node(
    nodes: &mut NodeAppend<'_>,
    primary: &str,
    index: usize,
    kind: NodeKind,
) -> SpiceResult<NodeId> {
    let name = format!("{primary}#passive{index}");
    if nodes.get(&name).is_some() {
        return Err(SpiceError::circuit(format!(
            "passive network node collision: {name}"
        )));
    }
    let id = nodes.intern(&name)?;
    nodes.set_kind(id, kind)?;
    Ok(id)
}
