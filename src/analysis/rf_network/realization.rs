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

#[derive(Debug, Clone, Copy)]
pub(crate) struct Edge {
    pub nodes: [usize; 2],
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
    pub(crate) edges: Vec<Edge>,
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
        if !valid || self.edges.len() >= 100_000 || self.nodes > 100_000 {
            return Err(invalid("unrepresentable or oversized passive realization"));
        }
        self.edges.push(Edge { nodes, element });
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

    /// Validation result for distributed-line approximations.
    #[must_use]
    pub const fn report(&self) -> Option<LadderReport> {
        self.report
    }

    /// Evaluate frequency-domain eligibility of a prepared realization.
    /// # Errors
    /// A changed preparation temperature or an out-of-band frequency.
    pub fn validate_context(&self, hz: f64, temperature: f64) -> SpiceResult<()> {
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
    fn new(mode: LineMode, length: f64, spec: LadderSpec) -> SpiceResult<Self> {
        let p = mode.parameters();
        let (dc, _) = mode.rlgc(0., spec.temperature)?;
        let factor = p
            .resistance_temperature_coefficient
            .mul_add(spec.temperature - p.reference_temperature, 1.);
        let mut skin = Vec::new();
        if p.skin != 0. && length != 0. {
            // sqrt(s) = (1/pi) integral s/(s+t) t^(-1/2) dt.
            // Logarithmic midpoint quadrature gives a Foster series of
            // parallel R-L branches with positive residues and rates.
            let [low, high] = spec.band.limits();
            let log_min = low.max(high * 1e-9).ln() + (TAU * 1e-8).ln();
            let log_max = high.ln() + (TAU * 1e8).ln();
            let step = (log_max - log_min) / 64.;
            for i in 0..64 {
                let rate = (f64::from(i).mul_add(step, 0.5_f64.mul_add(step, log_min))).exp();
                let resistance =
                    p.skin * factor.sqrt() * length * rate.sqrt() * step / (PI * PI.sqrt());
                if !rate.is_finite() || rate <= 0. || !resistance.is_finite() || resistance <= 0. {
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
            graph.edge([at, 0], Element::Capacitance(self.capacitance * weight))?;
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

fn validation_grid(band: FrequencyBand) -> Vec<f64> {
    let [low, high] = band.limits();
    let log_min = low.max(high * 1e-9).ln();
    let log_max = high.ln();
    let mut grid = vec![0., low, high];
    for i in 1..256 {
        grid.push((f64::from(i) / 256.).mul_add(high - low, low));
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
    grid
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
        let grid = validation_grid(spec.band);
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
        while sections <= spec.max_sections {
            let length = self.length
                / f64::from(u32::try_from(sections).map_err(|_| invalid("section count"))?);
            let cells = self
                .modes
                .iter()
                .map(|&mode| Cell::new(mode, length, spec))
                .collect::<SpiceResult<Vec<_>>>()?;
            let mut worst_error: f64 = 0.;
            for (cell, exact) in cells.iter().zip(&targets) {
                for (&hz, &target) in grid.iter().zip(exact) {
                    let approximation =
                        cell.waves(hz, spec.reference_impedance)?.repeat(sections)?;
                    worst_error = worst_error.max(approximation.error(target));
                }
            }
            if worst_error <= spec.max_error {
                return self.build_realization(
                    spec,
                    &cells,
                    LadderReport {
                        sections,
                        samples: grid.len(),
                        worst_error,
                    },
                );
            }
            sections *= 2;
        }
        Err(invalid(
            "ladder cannot meet scattering error within section budget",
        ))
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
