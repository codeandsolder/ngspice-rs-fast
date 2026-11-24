//! Positive-element, finite-band time-domain realization of prepared lines.
//! No fitting or frequency-local material substitution takes place in Newton.
use super::{FrequencyBand, LineMode, ModalRelation, PreparedLine, invalid, nonnegative, scaled};
use crate::primitives::{Complex, SpiceResult};
use std::f64::consts::{PI, TAU};

/// A passive lumped element in a prepared network.
#[derive(Debug, Clone, Copy)]
pub enum Element {
    /// Resistance in ohms (strictly positive).
    Resistance(f64),
    /// Inductance in henries (zero represents an exact short).
    Inductance(f64),
    /// Capacitance in farads.
    Capacitance(f64),
}

/// A lumped element attached to graph-local node indices.
#[derive(Debug, Clone, Copy)]
pub struct LumpedElement {
    /// Zero is the common reference; indices 1 through the port count
    /// are port nodes. Remaining indices refer to internal nodes.
    pub nodes: [usize; 2],
    /// Physical R, L or C value in SI units.
    pub element: Element,
}

/// An immutable positive-element graph with port voltages in modal coordinates.
///
/// Port nodes follow the common reference node (index zero). The real
/// orthogonal basis maps physical differential port voltages into those
/// coordinates; the transpose maps currents back, conserving power.
#[derive(Debug, Clone)]
pub struct PassiveRealization {
    pub(crate) ports: usize,
    pub(crate) nodes: usize,
    pub(crate) edges: Vec<LumpedElement>,
    pub(crate) basis: Vec<f64>,
    pub(crate) band: FrequencyBand,
    pub(crate) temperature: Option<f64>,
    report: Option<LadderReport>,
}

/// Error budget and resource bound for distributed-line approximation.
#[derive(Debug, Clone, Copy)]
pub struct LadderSpec {
    /// The frequency range over which the approximation is checked.
    pub band: FrequencyBand,
    /// Common real wave reference impedance in ohms.
    pub reference_impedance: f64,
    /// Maximum absolute error of either modal scattering eigenvalue.
    pub max_error: f64,
    /// Upper bound on spatial sections per mode (at most 256).
    pub max_sections: usize,
    /// Material temperature in degrees Celsius.
    pub temperature: f64,
}

/// Observed approximation error on the deterministic validation grid.
#[derive(Debug, Clone, Copy)]
pub struct LadderReport {
    /// Number of cascaded symmetric pi cells per mode.
    pub sections: usize,
    /// Number of skin-effect Foster branches per spatial section, summed
    /// across all modes. Zero means their omission met the error budget.
    pub skin_poles: usize,
    /// Number of distinct validation frequencies, including DC.
    pub samples: usize,
    /// Maximum observed scattering eigenvalue error.
    pub worst_error: f64,
}

impl PassiveRealization {
    fn empty(ports: usize, band: FrequencyBand, temperature: Option<f64>) -> Self {
        Self {
            ports,
            nodes: ports + 1,
            edges: Vec::new(),
            basis: (0..ports)
                .flat_map(|i| (0..ports).map(move |j| f64::from(u8::from(i == j))))
                .collect(),
            band,
            temperature,
            report: None,
        }
    }

    const fn node(&mut self) -> usize {
        let index = self.nodes;
        self.nodes += 1;
        index
    }

    fn edge(&mut self, nodes: [usize; 2], element: Element) -> SpiceResult<()> {
        let valid = match element {
            Element::Resistance(r) => r.is_finite() && r > 0. && r.recip().is_finite(),
            Element::Inductance(l) | Element::Capacitance(l) => nonnegative(l),
        };
        if !valid
            || nodes[0] == nodes[1]
            || nodes.iter().any(|&i| i >= self.nodes)
            || self.edges.len() >= 100_000
            || self.nodes > 100_000
        {
            return Err(invalid("unrepresentable or oversized passive realization"));
        }
        self.edges.push(LumpedElement { nodes, element });
        Ok(())
    }

    /// Import a positive-element discontinuity or passive rational model
    /// prepared outside the solver.
    ///
    /// Internal node indices must have a DC conductive path to a port or
    /// reference. The simulator never invents a leakage path for an
    /// otherwise undefined bias point.
    /// # Errors
    /// Invalid topology, values, DC connectivity or resource budget.
    pub fn lumped(
        ports: usize,
        internal_nodes: usize,
        elements: &[LumpedElement],
        band: FrequencyBand,
    ) -> SpiceResult<Self> {
        if !(1..=32).contains(&ports)
            || internal_nodes > 100_000 - ports - 1
            || elements.len() > 100_000
        {
            return Err(invalid("invalid lumped network dimensions"));
        }
        let mut graph = Self::empty(ports, band, None);
        graph.nodes += internal_nodes;
        for &edge in elements {
            graph.edge(edge.nodes, edge.element)?;
        }
        graph.validate_bias_connectivity()?;
        Ok(graph)
    }

    fn validate_bias_connectivity(&self) -> SpiceResult<()> {
        // Union-find over the DC conductive subgraph. Port and reference
        // nodes are anchors; ideal inductive cycles would make DC branch
        // currents indeterminate even with all terminal voltages fixed.
        const fn root(parent: &mut [usize], mut node: usize) -> usize {
            while parent[node] != node {
                parent[node] = parent[parent[node]];
                node = parent[node];
            }
            node
        }
        let mut parent: Vec<usize> = (0..self.nodes).collect();
        for inductive_only in [true, false] {
            for edge in &self.edges {
                let include = if inductive_only {
                    matches!(edge.element, Element::Inductance(_))
                } else {
                    matches!(edge.element, Element::Resistance(_))
                };
                if include {
                    let a = root(&mut parent, edge.nodes[0]);
                    let b = root(&mut parent, edge.nodes[1]);
                    if a == b && inductive_only {
                        return Err(invalid(
                            "inductive loop makes the DC branch equations singular",
                        ));
                    }
                    parent[a] = b;
                }
            }
        }
        let mut anchors = vec![false; self.nodes];
        for i in 0..=self.ports {
            anchors[root(&mut parent, i)] = true;
        }
        if (self.ports + 1..self.nodes).any(|i| !anchors[root(&mut parent, i)]) {
            return Err(invalid("lumped internal node has no conductive DC path"));
        }
        Ok(())
    }

    /// A passive lumped discontinuity model with series inductive arms and
    /// common shunt capacitance, for extracted bend, step or tee parameters.
    ///
    /// With one arm this represents an open-end capacitive correction.
    /// Zero-valued inductive arms are exact shorts.
    /// # Errors
    /// Invalid or unrepresentable elements, or unsupported port count.
    pub fn junction(arms: &[f64], capacitance: f64, band: FrequencyBand) -> SpiceResult<Self> {
        if arms.is_empty() || arms.len() > 16 || !nonnegative(capacitance) {
            return Err(invalid("invalid lumped RF junction"));
        }
        let mut graph = Self::empty(arms.len(), band, None);
        let center = graph.node();
        for (k, &l) in arms.iter().enumerate() {
            graph.edge([k + 1, center], Element::Inductance(l))?;
        }
        if capacitance > 0. {
            graph.edge([center, 0], Element::Capacitance(capacitance))?;
        }
        Ok(graph)
    }

    /// Number of differential physical ports.
    #[must_use]
    pub const fn ports(&self) -> usize {
        self.ports
    }

    /// Number of lumped circuit elements.
    #[must_use]
    pub const fn element_count(&self) -> usize {
        self.edges.len()
    }

    /// Borrow the graph-local elements for artifact serialization and inspection.
    #[must_use]
    pub fn elements(&self) -> &[LumpedElement] {
        &self.edges
    }

    /// Number of internal nodes, excluding reference and port nodes.
    #[must_use]
    pub const fn internal_node_count(&self) -> usize {
        self.nodes - self.ports - 1
    }

    /// Validation result for distributed-line approximations.
    #[must_use]
    pub const fn report(&self) -> Option<LadderReport> {
        self.report
    }

    /// Evaluate frequency-domain eligibility of a prepared realization.
    /// # Errors
    /// A changed preparation temperature or an out-of-band frequency.
    pub fn validate_context(&self, hz: f64, temperature: f64) -> SpiceResult<()> {
        if !temperature.is_finite() || temperature <= -273.15 {
            return Err(invalid("invalid realization temperature"));
        }
        if self
            .temperature
            .is_some_and(|t| t.to_bits() != temperature.to_bits())
        {
            return Err(invalid("realization temperature changed; prepare again"));
        }
        if hz != 0. && !self.band.contains(hz) {
            return Err(invalid("frequency outside realization band"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct SkinPole {
    resistance: f64,
    rate: f64,
}

struct Cell {
    resistance: f64,
    inductance: f64,
    conductance: f64,
    capacitance: f64,
    delta_c: f64,
    tau: f64,
    skin: Vec<SkinPole>,
}

impl Cell {
    fn new(mode: LineMode, length: f64, spec: LadderSpec, poles: u32) -> SpiceResult<Self> {
        let p = mode.parameters();
        let (dc, _) = mode.rlgc(0., spec.temperature)?;
        let factor = p
            .resistance_temperature_coefficient
            .mul_add(spec.temperature - p.reference_temperature, 1.);
        let mut skin = Vec::new();
        if p.skin != 0. && length != 0. && poles != 0 {
            // sqrt(s) = (1/pi) integral s/(s+t) t^(-1/2) dt.
            // Logarithmic midpoint quadrature gives a Foster series of
            // parallel R-L branches with positive residues and rates.
            let [low, high] = spec.band.limits();
            let log_min = low.max((high * 1e-9).max(f64::MIN_POSITIVE)).ln() + (TAU * 1e-8).ln();
            let log_max = high.ln() + (TAU * 1e8).ln();
            let step = (log_max - log_min) / f64::from(poles);
            for i in 0..poles {
                let rate = (f64::from(i).mul_add(step, 0.5_f64.mul_add(step, log_min))).exp();
                let resistance =
                    p.skin * factor.sqrt() * length * rate.sqrt() * step / (PI * PI.sqrt());
                let inductance = resistance / rate;
                if !rate.is_finite()
                    || rate <= 0.
                    || !resistance.is_finite()
                    || resistance <= 0.
                    || !inductance.is_finite()
                    || inductance <= 0.
                {
                    return Err(invalid("unrepresentable skin-effect quadrature"));
                }
                skin.push(SkinPole { resistance, rate });
            }
        }
        let cell = Self {
            resistance: dc.re * length,
            inductance: p.inductance * length,
            conductance: p.conductance * length,
            capacitance: p.capacitance * length,
            delta_c: p.relaxation_capacitance * length,
            tau: p.relaxation_time,
            skin,
        };
        if length > 0.
            && (cell.inductance == 0.
                || cell.capacitance == 0.
                || (p.resistance > 0. && cell.resistance == 0.)
                || (p.conductance > 0. && cell.conductance == 0.)
                || (p.relaxation_capacitance > 0. && cell.delta_c == 0.))
        {
            return Err(invalid("lumped cell coefficient underflow"));
        }
        Ok(cell)
    }

    fn element_count(&self, sections: usize) -> usize {
        let series = 1 + usize::from(self.resistance != 0.) + 2 * self.skin.len();
        let shunt = usize::from(self.conductance != 0.)
            + usize::from(self.capacitance != 0.)
            + 2 * usize::from(self.delta_c != 0.);
        sections * series + (sections + 1) * shunt
    }

    fn fit_error(
        &self,
        frequencies: &[f64],
        exact: &[[Complex; 2]],
        sections: usize,
        spec: LadderSpec,
    ) -> Option<f64> {
        let mut worst: f64 = 0.;
        for (&hz, &target) in frequencies.iter().zip(exact) {
            let approximation = self
                .waves(hz, spec.reference_impedance)
                .ok()?
                .repeat(sections)
                .ok()?;
            worst = worst.max(approximation.error(target));
            if worst > spec.max_error {
                return None;
            }
        }
        Some(worst)
    }

    fn series_shunt(&self, hz: f64) -> (Complex, Complex) {
        let omega = TAU * hz;
        let s = Complex::new(0., omega);
        let mut series = Complex::new(self.resistance, omega * self.inductance);
        for pole in &self.skin {
            series = series + scaled(s / (s + Complex::new(pole.rate, 0.)), pole.resistance);
        }
        let mut shunt = Complex::new(self.conductance, omega * self.capacitance);
        if self.delta_c != 0. {
            shunt = shunt
                + scaled(
                    s / (Complex::new(1., 0.) + scaled(s, self.tau)),
                    self.delta_c,
                );
        }
        (series, shunt)
    }

    fn waves(&self, hz: f64, reference: f64) -> SpiceResult<Waves> {
        let (z, y) = self.series_shunt(hz);
        let z = scaled(z, reference.recip());
        let y = scaled(y, reference);
        let zy = z * y;
        let a = Complex::new(1., 0.) + scaled(zy, 0.5);
        let c = y * (Complex::new(1., 0.) + scaled(zy, 0.25));
        let denominator = scaled(a, 2.) + z + c;
        let reflection = (z - c) / denominator;
        let transmission = Complex::new(2., 0.) / denominator;
        Waves::checked(reflection, transmission)
    }

    fn shunt(&self, graph: &mut PassiveRealization, at: usize, weight: f64) -> SpiceResult<()> {
        if self.conductance != 0. {
            graph.edge(
                [at, 0],
                Element::Resistance((self.conductance * weight).recip()),
            )?;
        }
        if self.capacitance != 0. {
            let capacitance = self.capacitance * weight;
            if capacitance == 0. {
                return Err(invalid("shunt capacitance underflow"));
            }
            graph.edge([at, 0], Element::Capacitance(capacitance))?;
        }
        if self.delta_c != 0. {
            let middle = graph.node();
            let cap = self.delta_c * weight;
            graph.edge([at, middle], Element::Resistance(self.tau / cap))?;
            graph.edge([middle, 0], Element::Capacitance(cap))?;
        }
        Ok(())
    }

    fn connect(&self, graph: &mut PassiveRealization, start: usize, end: usize) -> SpiceResult<()> {
        let mut from = start;
        if self.resistance != 0. {
            let next = graph.node();
            graph.edge([from, next], Element::Resistance(self.resistance))?;
            from = next;
        }
        for pole in &self.skin {
            let next = graph.node();
            graph.edge([from, next], Element::Resistance(pole.resistance))?;
            graph.edge(
                [from, next],
                Element::Inductance(pole.resistance / pole.rate),
            )?;
            from = next;
        }
        graph.edge([from, end], Element::Inductance(self.inductance))
    }

    fn expand(
        &self,
        graph: &mut PassiveRealization,
        start: usize,
        end: usize,
        count: usize,
    ) -> SpiceResult<()> {
        let mut at = start;
        self.shunt(graph, at, 0.5)?;
        for i in 0..count {
            let next = if i + 1 == count { end } else { graph.node() };
            self.connect(graph, at, next)?;
            self.shunt(graph, next, if i + 1 == count { 0.5 } else { 1. })?;
            at = next;
        }
        Ok(())
    }
}

// Cascading identical reciprocal symmetric cells preserves symmetry.
// Work with bounded scattering coefficients rather than exponential ABCD.
#[derive(Clone, Copy)]
struct Waves {
    reflection: Complex,
    transmission: Complex,
}

impl Waves {
    fn checked(reflection: Complex, transmission: Complex) -> SpiceResult<Self> {
        if !reflection.is_finite() || !transmission.is_finite() {
            return Err(invalid("unrepresentable ladder scattering"));
        }
        Ok(Self {
            reflection,
            transmission,
        })
    }

    fn double(self) -> SpiceResult<Self> {
        let den = Complex::new(1., 0.) - self.reflection * self.reflection;
        let through = self.transmission * self.transmission / den;
        Self::checked(self.reflection + through * self.reflection, through)
    }

    fn repeat(mut self, count: usize) -> SpiceResult<Self> {
        for _ in 0..count.trailing_zeros() {
            self = self.double()?;
        }
        Ok(self)
    }

    fn error(self, exact: [Complex; 2]) -> f64 {
        ((self.reflection + self.transmission) - exact[0])
            .magnitude()
            .max(((self.reflection - self.transmission) - exact[1]).magnitude())
    }
}

fn validation_grid(band: FrequencyBand, phase: f64) -> SpiceResult<Vec<f64>> {
    let [low, high] = band.limits();
    let log_min = low.max((high * 1e-9).max(f64::MIN_POSITIVE)).ln();
    let log_max = high.ln();
    // At least sixteen linear samples per full electrical wavelength,
    // in addition to logarithmic sampling of material corner frequencies.
    // Explicitly reject a line too long for the bounded validation work.
    let required = 16. * phase / TAU;
    if !required.is_finite() || required > 16_384. {
        return Err(invalid(
            "electrical length exceeds ladder validation grid budget",
        ));
    }
    let intervals = (8..=14)
        .map(|shift| 1_u32 << shift)
        .find(|&count| f64::from(count) >= required)
        .ok_or_else(|| invalid("electrical length exceeds ladder validation grid budget"))?;
    let mut grid = vec![0., low, high];
    for i in 1..intervals {
        grid.push((f64::from(i) / f64::from(intervals)).mul_add(high - low, low));
    }
    for i in 1..128 {
        grid.push(
            (f64::from(i) / 128.)
                .mul_add(log_max - log_min, log_min)
                .exp()
                .clamp(low, high),
        );
    }
    grid.sort_by(f64::total_cmp);
    grid.dedup_by(|a, b| a.to_bits() == b.to_bits());
    Ok(grid)
}

fn fit_cell(
    mode: LineMode,
    length: f64,
    spec: LadderSpec,
    sections: usize,
    grid: &[f64],
    target: &[[Complex; 2]],
) -> Option<(Cell, f64)> {
    let poles: &[u32] = if mode.parameters().skin == 0. {
        &[0]
    } else {
        &[0, 4, 8, 16, 32, 64]
    };
    for &count in poles {
        if let Ok(cell) = Cell::new(mode, length, spec, count)
            && let Some(error) = cell.fit_error(grid, target, sections, spec)
        {
            return Some((cell, error));
        }
    }
    None
}

struct LadderCandidate {
    cells: Vec<Cell>,
    report: LadderReport,
    elements: usize,
}

impl PreparedLine {
    /// Prepare a causal finite-dimensional passive approximation.
    ///
    /// The reported error is an observed grid maximum, not a mathematical
    /// continuous-band certificate. Transient excitation must be band-limited
    /// to the useful range. Every resulting element is an ordinary R/L/C.
    /// # Errors
    /// Invalid bounds, unsupported range, or exhausted error/resource budget.
    pub fn realize(&self, spec: LadderSpec) -> SpiceResult<PassiveRealization> {
        if self.dispersion.iter().any(Option::is_some) {
            return Err(invalid(
                "empirical dispersion requires an independently prepared causal realization",
            ));
        }
        let [low, high] = spec.band.limits();
        let [valid_low, valid_high] = self.band.limits();
        if !high.is_normal()
            || high > f64::MAX / TAU
            || high <= 0.
            || high > valid_high
            || low < valid_low
            || !spec.reference_impedance.is_finite()
            || spec.reference_impedance <= 0.
            || !spec.max_error.is_finite()
            || spec.max_error <= 0.
            || spec.max_error >= 1.
            || !(1..=256).contains(&spec.max_sections)
        {
            return Err(invalid("invalid ladder specification"));
        }
        let phase = self.modes.iter().try_fold(0_f64, |max, mode| {
            let gamma = mode.propagation(high, spec.temperature)?.gamma;
            Ok::<_, crate::primitives::SpiceError>(max.max(gamma.im * self.length))
        })?;
        let grid = validation_grid(spec.band, phase)?;
        let targets = self
            .modes
            .iter()
            .map(|&mode| {
                grid.iter()
                    .map(|&hz| {
                        let (z, y) = mode.rlgc(hz, spec.temperature)?;
                        ModalRelation::new(z, y, self.length)?.reflections(spec.reference_impedance)
                    })
                    .collect::<SpiceResult<Vec<_>>>()
            })
            .collect::<SpiceResult<Vec<_>>>()?;
        let mut sections = 1;
        let mut best: Option<LadderCandidate> = None;
        while sections <= spec.max_sections {
            if best
                .as_ref()
                .is_some_and(|old| sections * self.conductors() >= old.elements)
            {
                break;
            }
            if let Some(candidate) = self.fit_sections(spec, sections, &grid, &targets)?
                && best
                    .as_ref()
                    .is_none_or(|old| candidate.elements < old.elements)
            {
                best = Some(candidate);
            }
            sections *= 2;
        }
        let best = best
            .ok_or_else(|| invalid("ladder cannot meet scattering error within section budget"))?;
        self.build_realization(spec, &best.cells, best.report)
    }

    fn fit_sections(
        &self,
        spec: LadderSpec,
        sections: usize,
        grid: &[f64],
        targets: &[Vec<[Complex; 2]>],
    ) -> SpiceResult<Option<LadderCandidate>> {
        let length =
            self.length / f64::from(u32::try_from(sections).map_err(|_| invalid("section count"))?);
        if self.length > 0. && length == 0. {
            return Ok(None);
        }
        let mut cells = Vec::new();
        let mut worst_error: f64 = 0.;
        let mut elements = 0;
        let mut skin_poles = 0;
        for (&mode, target) in self.modes.iter().zip(targets) {
            let Some((cell, error)) = fit_cell(mode, length, spec, sections, grid, target) else {
                return Ok(None);
            };
            worst_error = worst_error.max(error);
            elements += cell.element_count(sections);
            skin_poles += cell.skin.len();
            cells.push(cell);
        }
        if elements > 100_000 {
            return Ok(None);
        }
        Ok(Some(LadderCandidate {
            cells,
            elements,
            report: LadderReport {
                sections,
                skin_poles,
                samples: grid.len(),
                worst_error,
            },
        }))
    }

    fn build_realization(
        &self,
        spec: LadderSpec,
        cells: &[Cell],
        report: LadderReport,
    ) -> SpiceResult<PassiveRealization> {
        let n = self.conductors();
        let mut graph = PassiveRealization::empty(2 * n, spec.band, Some(spec.temperature));
        graph.basis.fill(0.);
        for i in 0..n {
            for j in 0..n {
                graph.basis[i * 2 * n + j] = self.basis[i * n + j];
                graph.basis[(i + n) * 2 * n + j + n] = self.basis[i * n + j];
            }
        }
        for (k, cell) in cells.iter().enumerate() {
            cell.expand(&mut graph, k + 1, k + n + 1, report.sections)?;
        }
        graph.report = Some(report);
        Ok(graph)
    }
}
