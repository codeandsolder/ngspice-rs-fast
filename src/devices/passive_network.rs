//! Physical two-terminal storage networks, bound to ordinary native RLC
//! devices.
//!
//! Source-dependent defaults and current observation conventions are
//! resolved separately; the winding, storage and terminal identities are
//! intentionally distinct.

mod damping;
pub(crate) mod report;
use damping::DampingShunt;
pub use damping::{AnalysisDamping, ConductanceLaw};
pub(super) use report::CurrentReportSpec;
pub(crate) use report::{BoundCurrentReport, CurrentOutputLaw, ReportPhase};
pub use report::{CurrentReference, CurrentReporting, CurrentReportingLaw};

use crate::devices::{
    Capacitor, Circuit, Device, IndependentSource, Inductor, Resistor, ResistorNoise, StorageKind,
    elaboration::{ElaborationSize, reserve},
    passive_bias::PassiveInitialization,
    rlc::REACTIVE_STATE_COUNT,
    sources::{SourceExcitation, SourceInputs},
};
use crate::primitives::{NodeAppend, NodeId, NodeKind, Real, SpiceError, SpiceResult};

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
    /// Analysis-dependent damping directly across the series inductance.
    pub series_inductance_damping: Option<AnalysisDamping>,
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
    /// Analysis-dependent damping directly across the main storage.
    pub storage_damping: Option<AnalysisDamping>,
    /// Analysis-dependent damping across the external terminal pair.
    pub terminal_damping: Option<AnalysisDamping>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    SeriesResistance,
    SeriesInductance,
    SeriesInductanceShunt,
    SeriesInductanceDamping,
    Primary,
    StorageShunt,
    ParallelResistance,
    ParallelCapacitance,
    StorageDamping,
    TerminalDamping,
}

impl Role {
    fn name(self, primary: &str) -> String {
        match self {
            Self::SeriesResistance => format!("r{primary}#series"),
            Self::SeriesInductance => format!("l{primary}#series"),
            Self::SeriesInductanceShunt => format!("r{primary}#series_shunt"),
            Self::SeriesInductanceDamping => format!("r{primary}#series_damping"),
            Self::Primary => primary.to_owned(),
            Self::StorageShunt => format!("r{primary}#storage_shunt"),
            Self::ParallelResistance => format!("r{primary}#parallel"),
            Self::ParallelCapacitance => format!("c{primary}#parallel"),
            Self::StorageDamping => format!("r{primary}#storage_damping"),
            Self::TerminalDamping => format!("r{primary}#terminal_damping"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Element {
    Resistance(Real),
    Storage(StorageKind, Real),
    Damping(AnalysisDamping),
}

impl Element {
    const fn state_slots(self) -> usize {
        match self {
            Self::Resistance(_) | Self::Damping(_) => 0,
            Self::Storage(..) => REACTIVE_STATE_COUNT,
        }
    }

    fn scaled(self, multiplicity: Real) -> SpiceResult<Self> {
        let (original, scaled) = match self {
            Self::Resistance(r) => (r, Self::Resistance(r / multiplicity)),
            Self::Damping(damping) => return damping.scaled(multiplicity).map(Self::Damping),
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
            Self::Damping(_) => return Err(SpiceError::circuit("unexpected scalar damping")),
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
    reporting: Option<CurrentReporting>,
    control: Option<CurrentReference>,
    primary_probe: bool,
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
        input: &PassiveNetworkCoefficients,
        initialization: Option<PassiveInitialization>,
    ) -> SpiceResult<Option<Self>> {
        Self::validate(input, initialization.as_ref())?;
        if input.multiplicity == 0. {
            return Ok(None);
        }

        let mut edges = Vec::new();
        reserve(&mut edges, 10, None)?;
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
            if let Some(damping) = input.series_inductance_damping {
                append(
                    Role::SeriesInductanceDamping,
                    Element::Damping(damping),
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
        if let Some(damping) = input.storage_damping {
            append(
                Role::StorageDamping,
                Element::Damping(damping),
                [positive, 1],
            )?;
        }
        if let Some(damping) = input.terminal_damping {
            append(Role::TerminalDamping, Element::Damping(damping), [0, 1])?;
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
            reporting: None,
            control: None,
            primary_probe: false,
        }))
    }

    fn validate(
        input: &PassiveNetworkCoefficients,
        initialization: Option<&PassiveInitialization>,
    ) -> SpiceResult<()> {
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
        if let Some(policy) = initialization {
            policy.check_kind(input.kind)?;
        }
        Ok(())
    }

    /// Install an explicit current-reporting policy without changing the
    /// network's electrical equations.
    ///
    /// # Errors
    /// Nonfinite reporting coefficients.
    pub fn with_reporting(mut self, reporting: CurrentReporting) -> SpiceResult<Self> {
        self.reporting = Some(reporting.validate()?);
        self.update_primary_probe();
        Ok(self)
    }

    /// Select the physical branch used by controlling-current references.
    #[must_use]
    pub fn with_current_control(mut self, reference: CurrentReference) -> Self {
        self.control = Some(reference);
        self.update_primary_probe();
        self
    }

    fn update_primary_probe(&mut self) {
        self.primary_probe = (self.reporting.is_some_and(CurrentReporting::needs_primary)
            || self.control == Some(CurrentReference::Primary))
            && self.edges.iter().any(|edge| {
                edge.role == Role::Primary
                    && matches!(edge.element, Element::Storage(StorageKind::Capacitor, _))
            });
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
        circuit
            .add_prepared_devices(self.size(), |nodes| self.build(nodes, &binding, terminals))?;
        Ok(binding)
    }

    pub(super) fn size(&self) -> ElaborationSize {
        ElaborationSize::new(
            self.edges.len() + 1 + usize::from(self.primary_probe),
            self.nodes - 1 + usize::from(self.primary_probe),
            self.edges
                .iter()
                .map(|edge| edge.element.state_slots())
                .sum(),
        )
    }

    pub(super) fn build(
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
        reserve(
            &mut devices,
            self.edges.len() + 1 + usize::from(self.primary_probe),
            None,
        )?;
        let primary = if self.primary_probe {
            format!("v{}#primary", binding.primary)
        } else {
            binding.primary.clone()
        };
        let report =
            (self.reporting.is_some() || self.control.is_some()).then(|| CurrentReportSpec {
                original: binding.primary.clone(),
                primary: primary.clone(),
                terminals,
                reporting: self.reporting,
                control: self.control,
            });
        devices.push(Box::new(IndependentSource::current_probe(
            &binding.terminal_probe,
            [positive, alias],
            report,
        )?));
        for edge in &self.edges {
            let name = edge.role.name(&binding.primary);
            let mut terminals = edge.nodes.map(|index| ids[index]);
            if self.primary_probe && edge.role == Role::Primary {
                let sense = new_node(nodes, &binding.primary, self.nodes, NodeKind::ProbeAlias)?;
                devices.push(Box::new(IndependentSource::current_probe(
                    &primary,
                    [terminals[0], sense],
                    None,
                )?));
                terminals[0] = sense;
            }
            let mut device: Box<dyn Device> = match edge.element {
                Element::Resistance(r) => Box::new(Resistor::new(name, terminals, r)?.with_noise(
                    ResistorNoise {
                        temperature: self.temperature,
                        ..Default::default()
                    },
                )),
                Element::Damping(damping) => Box::new(DampingShunt {
                    name,
                    terminals,
                    damping,
                    temperature: self.temperature,
                }),
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

/// Explicit physical output impedance of an independent voltage source.
///
/// The shunt capacitance is internal to the terminal current probe; the
/// series impedance carries both the load and capacitive currents.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VoltageNetworkCoefficients {
    /// Signed series resistance in ohms (zero denotes an exact wire).
    pub series_resistance: Real,
    /// Capacitance across the output pair, before the terminal current probe.
    pub parallel_capacitance: Real,
    /// Additional capacitance directly across the series resistance.
    pub series_shunt_capacitance: Real,
    /// Resolved series-resistor noise temperature, in Celsius.
    pub temperature: Option<Real>,
}

/// Prepared independent voltage source and its physical output impedance.
///
/// The excitation identity is unchanged for sweeps and input-referred noise;
/// the separately named probe gives current into the external positive port.
#[derive(Debug, Clone)]
pub struct PreparedVoltageNetwork {
    coefficients: VoltageNetworkCoefficients,
    excitation: SourceExcitation,
    reporting: Option<CurrentReporting>,
    control: Option<CurrentReference>,
}

/// Identities for voltage forcing and external current reporting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoltageNetworkBinding {
    /// Ideal voltage excitation and its internal branch current.
    pub excitation: String,
    /// External terminal current (excluding the internal shunt capacitance).
    pub terminal_probe: String,
}

impl PreparedVoltageNetwork {
    /// Resolve voltage excitation and physical output coefficients.
    ///
    /// # Errors
    /// Invalid excitation, nonfinite coefficients or temperature, negative
    /// capacitance, or a nonzero resistance with nonfinite reciprocal.
    pub fn prepare(
        coefficients: VoltageNetworkCoefficients,
        inputs: SourceInputs,
    ) -> SpiceResult<Self> {
        if !coefficients.series_resistance.is_finite()
            || (coefficients.series_resistance != 0.
                && !(1. / coefficients.series_resistance).is_finite())
            || !coefficients.parallel_capacitance.is_finite()
            || coefficients.parallel_capacitance < 0.
            || !coefficients.series_shunt_capacitance.is_finite()
            || coefficients.series_shunt_capacitance < 0.
            || coefficients
                .temperature
                .is_some_and(|t| !t.is_finite() || t <= -273.15)
        {
            return Err(SpiceError::circuit(
                "invalid voltage output network coefficients",
            ));
        }
        Ok(Self {
            coefficients,
            excitation: SourceExcitation::from_inputs(true, inputs)?,
            reporting: None,
            control: None,
        })
    }

    /// Install an explicit current-reporting law, preserving forcing identity.
    ///
    /// # Errors
    /// Nonfinite reporting coefficients.
    pub fn with_reporting(mut self, reporting: CurrentReporting) -> SpiceResult<Self> {
        self.reporting = Some(reporting.validate()?);
        Ok(self)
    }

    /// Select the physical branch used by controlling-current references.
    #[must_use]
    pub const fn with_current_control(mut self, reference: CurrentReference) -> Self {
        self.control = Some(reference);
        self
    }

    /// Atomically bind the voltage source and its output impedance.
    ///
    /// # Errors
    /// Unknown terminals, duplicate names or exhausted elaboration limits.
    pub fn bind(
        &self,
        circuit: &mut Circuit,
        name: &str,
        terminals: [NodeId; 2],
    ) -> SpiceResult<VoltageNetworkBinding> {
        if name.is_empty()
            || terminals
                .iter()
                .any(|&id| circuit.nodes().node(id).is_none())
        {
            return Err(SpiceError::circuit("invalid voltage network binding"));
        }
        let binding = VoltageNetworkBinding {
            excitation: name.to_owned(),
            terminal_probe: format!("v{name}#port"),
        };
        circuit
            .add_prepared_devices(self.size(), |nodes| self.build(nodes, &binding, terminals))?;
        Ok(binding)
    }

    pub(super) fn size(&self) -> ElaborationSize {
        let series = usize::from(self.coefficients.series_resistance != 0.);
        let shunt = usize::from(self.coefficients.parallel_capacitance != 0.)
            + usize::from(series != 0 && self.coefficients.series_shunt_capacitance != 0.);
        ElaborationSize::new(2 + series + shunt, 1 + series, shunt * REACTIVE_STATE_COUNT)
    }

    pub(super) fn build(
        &self,
        nodes: &mut NodeAppend<'_>,
        binding: &VoltageNetworkBinding,
        [positive, negative]: [NodeId; 2],
    ) -> SpiceResult<Vec<Box<dyn Device>>> {
        let name = &binding.excitation;
        let output = new_node(nodes, name, 0, NodeKind::ProbeAlias)?;
        let series = self.coefficients.series_resistance;
        let source = if series == 0. {
            output
        } else {
            new_node(nodes, name, 1, NodeKind::Internal)?
        };
        let mut devices: Vec<Box<dyn Device>> = Vec::new();
        reserve(&mut devices, self.size().devices, None)?;
        devices.push(Box::new(
            self.excitation.clone().bind(name, [source, negative]),
        ));
        let report =
            (self.reporting.is_some() || self.control.is_some()).then(|| CurrentReportSpec {
                original: name.clone(),
                primary: name.clone(),
                terminals: [positive, negative],
                reporting: self.reporting,
                control: self.control,
            });
        devices.push(Box::new(IndependentSource::current_probe(
            &binding.terminal_probe,
            [positive, output],
            report,
        )?));
        if series != 0. {
            devices.push(Box::new(
                Resistor::new(Role::SeriesResistance.name(name), [source, output], series)?
                    .with_noise(ResistorNoise {
                        temperature: self.coefficients.temperature,
                        absolute_thermal_conductance: true,
                        ..Default::default()
                    }),
            ));
        }
        if series != 0. && self.coefficients.series_shunt_capacitance != 0. {
            devices.push(Box::new(Capacitor::new(
                format!("c{name}#series_shunt"),
                [source, output],
                self.coefficients.series_shunt_capacitance,
                None,
            )?));
        }
        if self.coefficients.parallel_capacitance != 0. {
            devices.push(Box::new(Capacitor::new(
                Role::ParallelCapacitance.name(name),
                [output, negative],
                self.coefficients.parallel_capacitance,
                None,
            )?));
        }
        Ok(devices)
    }
}

/// Parsed explicit network after lexical expansion and numeric literalization.
/// A removed storage instance still reserves its written instance identity,
/// but contributes no devices or private nodes.
pub(super) enum NetworkPlan {
    Storage(Option<PreparedPassiveNetwork>),
    Voltage(Box<PreparedVoltageNetwork>),
}

impl NetworkPlan {
    pub(super) fn prepare(
        instance: &crate::netlist::ast::DeviceInstance,
    ) -> SpiceResult<Option<Self>> {
        use crate::netlist::ast::ParameterKind;

        if !instance
            .parameters
            .iter()
            .any(|p| crate::netlist::network_parameter(instance.designator, &p.name))
        {
            return Ok(None);
        }
        let invalid = |message| SpiceError::parse(instance.location.clone(), message);
        if instance.model.is_some() || instance.nodes.len() != 2 {
            return Err(invalid(
                "physical network requires a resolved two-terminal instance",
            ));
        }

        // All network-specific setters are finite scalar coefficients after
        // occurrence expansion. Duplicate setters keep normal last-wins order.
        let mut values = std::collections::BTreeMap::new();
        let mut ordinary = instance.clone();
        ordinary.parameters.clear();
        for p in &instance.parameters {
            if crate::netlist::network_parameter(instance.designator, &p.name) {
                if p.kind != ParameterKind::Scalar {
                    return Err(SpiceError::parse(
                        p.location.clone(),
                        "physical network requires a resolved scalar coefficient",
                    ));
                }
                values.insert(p.name.as_str(), super::factory::scalar_number(p)?);
            } else {
                ordinary.parameters.push(p.clone());
            }
        }
        let get = |key| values.get(key).copied();
        let zero = |key| get(key).unwrap_or(0.);
        if instance.designator == 'v' {
            let mut network = PreparedVoltageNetwork::prepare(
                VoltageNetworkCoefficients {
                    series_resistance: zero("pn_rseries"),
                    parallel_capacitance: zero("pn_cparallel"),
                    series_shunt_capacitance: zero("pn_cseries_shunt"),
                    temperature: get("pn_temp"),
                },
                super::factory::voltage_network_inputs(&ordinary)?,
            )?;
            if let Some(reporting) = report::parse(&values, 0.)? {
                network = network.with_reporting(reporting)?;
            }
            if let Some(control) = report::parse_control(&values)? {
                network = network.with_current_control(control);
            }
            return Ok(Some(Self::Voltage(Box::new(network))));
        }

        Self::storage(&ordinary, &values).map(Some)
    }

    fn storage(
        instance: &crate::netlist::ast::DeviceInstance,
        values: &std::collections::BTreeMap<&str, Real>,
    ) -> SpiceResult<Self> {
        use crate::netlist::ast::ParameterKind;

        let invalid = |message| SpiceError::parse(instance.location.clone(), message);
        let get = |key| values.get(key).copied();
        let zero = |key| get(key).unwrap_or(0.);
        let (kind, primary) = match instance.designator {
            'c' => (StorageKind::Capacitor, "capacitance"),
            'l' => (StorageKind::Inductor, "inductance"),
            _ => return Err(invalid("invalid physical storage family")),
        };
        let mut value = None;
        let mut temperature = None;
        let mut multiplicity = 1.;
        let mut initial = None;
        for p in &instance.parameters {
            if p.kind != ParameterKind::Scalar {
                return Err(SpiceError::parse(
                    p.location.clone(),
                    "physical network requires resolved scalar storage",
                ));
            }
            let scalar = super::factory::scalar_number(p)?;
            match p.name.as_str() {
                name if name == primary => value = Some(scalar),
                "m" => multiplicity = scalar,
                "temp" => temperature = Some(scalar),
                "ic" => initial = Some(scalar),
                _ => {
                    return Err(SpiceError::parse(
                        p.location.clone(),
                        format!("unresolved physical network setter {}", p.name),
                    ));
                }
            }
        }
        let value = value.ok_or_else(|| invalid("physical network requires main storage value"))?;
        let coefficients = PassiveNetworkCoefficients {
            kind,
            value,
            series_resistance: zero("pn_rseries"),
            series_inductance: zero("pn_lseries"),
            series_inductance_shunt: get("pn_rseries_shunt").filter(|r| *r != 0.),
            series_inductance_damping: Self::damping(values, "series")?,
            storage_shunt: get("pn_rstorage_shunt").filter(|r| *r != 0.),
            parallel_resistance: get("pn_rparallel").filter(|r| *r != 0.),
            parallel_capacitance: zero("pn_cparallel"),
            multiplicity,
            temperature,
            storage_damping: Self::damping(values, "storage")?,
            terminal_damping: Self::damping(values, "terminal")?,
        };
        let initialization = Self::initialization(kind, initial, multiplicity, values)?;
        let Some(mut network) = PreparedPassiveNetwork::prepare(&coefficients, initialization)?
        else {
            return Ok(Self::Storage(None));
        };
        let conductance = network
            .edges
            .iter()
            .find_map(|edge| match (edge.role, edge.element) {
                (Role::ParallelResistance, Element::Resistance(r)) => Some(1. / r),
                _ => None,
            })
            .unwrap_or(0.);
        if let Some(reporting) = report::parse(values, conductance)? {
            network = network.with_reporting(reporting)?;
        }
        if let Some(control) = report::parse_control(values)? {
            network = network.with_current_control(control);
        }
        Ok(Self::Storage(Some(network)))
    }

    fn damping(
        coefficients: &std::collections::BTreeMap<&str, Real>,
        site: &str,
    ) -> SpiceResult<Option<AnalysisDamping>> {
        let get = |kind: &str, suffix: &str| {
            coefficients
                .get(format!("pn_g{site}_{kind}{suffix}").as_str())
                .copied()
        };
        let law = |kind| ConductanceLaw {
            constant: get(kind, "").unwrap_or(0.),
            gmin_scale: get(kind, "_gmin").unwrap_or(0.),
        };
        let noisy = get("noise", "").unwrap_or(0.);
        if noisy != 0. && noisy.to_bits() != 1_f64.to_bits() {
            return Err(SpiceError::circuit("damping noise selector must be 0 or 1"));
        }
        let damping = AnalysisDamping {
            steady: law("dc"),
            small_signal: law("ac"),
            transient: law("tran"),
            noisy: noisy != 0.,
        };
        Ok((damping != AnalysisDamping::default()).then_some(damping))
    }

    fn initialization(
        kind: StorageKind,
        initial: Option<Real>,
        multiplicity: Real,
        coefficients: &std::collections::BTreeMap<&str, Real>,
    ) -> SpiceResult<Option<PassiveInitialization>> {
        use super::passive_bias::PassiveBias;
        use crate::primitives::AnalysisKind;
        let initial = initial.map(|value| match kind {
            StorageKind::Capacitor => value,
            StorageKind::Inductor => value * multiplicity,
        });
        let mut constraints = Vec::new();
        for (analysis, key) in [
            (AnalysisKind::OperatingPoint, "pn_bias_op"),
            (AnalysisKind::DcSweep, "pn_bias_dc"),
            (AnalysisKind::Ac, "pn_bias_ac"),
            (AnalysisKind::Transient, "pn_bias_tran"),
        ] {
            let Some(&g) = coefficients.get(key) else {
                continue;
            };
            let target = initial.ok_or_else(|| {
                SpiceError::circuit("physical bias requires an explicit storage IC")
            })?;
            if !g.is_finite() || g < 0. {
                return Err(SpiceError::circuit("invalid physical bias conductance"));
            }
            let conductance = g * multiplicity;
            let bias = match kind {
                StorageKind::Capacitor => PassiveBias::Voltage {
                    voltage: target,
                    conductance: (g != 0.).then_some(conductance),
                },
                StorageKind::Inductor => PassiveBias::Current {
                    current: target,
                    conductance,
                },
            };
            if multiplicity != 0. {
                constraints.push((analysis, bias));
            }
        }
        if initial.is_none() && constraints.is_empty() {
            Ok(None)
        } else {
            PassiveInitialization::new(kind, initial, constraints).map(Some)
        }
    }

    pub(super) fn size(&self) -> ElaborationSize {
        match self {
            Self::Storage(Some(network)) => network.size(),
            Self::Storage(None) => ElaborationSize::default(),
            Self::Voltage(network) => network.size(),
        }
    }

    pub(super) fn build(
        &self,
        instance: &crate::netlist::ast::DeviceInstance,
        nodes: &mut NodeAppend<'_>,
    ) -> SpiceResult<Vec<Box<dyn Device>>> {
        let terminals = nodes.intern_array::<2>(&instance.nodes)?;
        let name = &instance.name;
        match self {
            Self::Storage(Some(network)) => network.build(
                nodes,
                &PassiveNetworkBinding {
                    primary: name.clone(),
                    terminal_probe: format!("v{name}#port"),
                },
                terminals,
            ),
            Self::Storage(None) => Ok(Vec::new()),
            Self::Voltage(network) => network.build(
                nodes,
                &VoltageNetworkBinding {
                    excitation: name.clone(),
                    terminal_probe: format!("v{name}#port"),
                },
                terminals,
            ),
        }
    }
}
