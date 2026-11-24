//! Noise conversion around a validated periodic orbit.
//!
//! A stationary independent generator at a signed input frequency may
//! inject a coherent column of amplitudes at many output sidebands.
//! Covariance is accumulated from these factors, never from independent
//! per-sideband output powers of a single correlated source.

use super::periodic_small_signal::{PeriodicResponse, PortChannel, Term};
use crate::devices::{
    AnalysisMode, LoadRequest, ModelContext, PreparedRun, StateSnapshot,
    noise::{BOLTZMANN, NoisePlan},
};
use crate::maths::Vector;
use crate::primitives::{Complex, ExactSum, Real, SpiceError, SpiceResult, filled_vec};

fn failure(message: &str) -> SpiceError {
    SpiceError::Numerical {
        context: "periodic noise".into(),
        message: message.into(),
    }
}

fn product(a: usize, b: usize, limit: usize, message: &str) -> SpiceResult<usize> {
    a.checked_mul(b)
        .filter(|&n| n <= limit)
        .ok_or_else(|| failure(message))
}

/// Explicit admission limits for selected covariance and native source sampling.
#[derive(Debug, Clone, Copy)]
pub struct NoiseLimits {
    /// Maximum number of requested output observations.
    pub outputs: usize,
    /// Maximum number of selected covariance pairs (including diagonals).
    pub pairs: usize,
    /// Maximum number of expanded adjoint cells retained.
    pub adjoint_entries: usize,
    /// Maximum independent generator factors accumulated.
    pub factors: usize,
    /// Maximum total sparse source/observation terms supplied explicitly.
    pub terms: usize,
    /// Maximum phase synthesis, physical orbit and source workspace cells.
    pub phase_entries: usize,
    /// Maximum phase/frequency model evaluations.
    pub evaluations: usize,
    /// Maximum scalar phase synthesis and source propagation work.
    pub work: usize,
}

impl Default for NoiseLimits {
    fn default() -> Self {
        Self {
            outputs: 256,
            pairs: 4096,
            adjoint_entries: 1_000_000,
            factors: 1_000_000,
            terms: 8_000_000,
            phase_entries: 8_000_000,
            evaluations: 100_000,
            work: 250_000_000,
        }
    }
}

/// One covariance observation, E[y_left conj(y_right)] per hertz.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CovariancePair {
    /// Left output ordinal.
    pub left: usize,
    /// Conjugated right output ordinal.
    pub right: usize,
}

/// Selected output spectral densities in caller-specified pair order.
pub struct NoiseResult {
    /// Corresponding output pairs.
    pub pairs: Vec<CovariancePair>,
    /// Complex covariance values in observation units squared per hertz.
    pub densities: Vec<Complex>,
    /// Number of independent generator columns accumulated.
    pub factors: usize,
}

#[derive(Clone)]
struct CovarianceSum {
    real: ExactSum,
    imaginary: ExactSum,
}

impl CovarianceSum {
    fn new() -> Self {
        Self {
            real: ExactSum::new(),
            imaginary: ExactSum::new(),
        }
    }

    fn add(&mut self, a: Complex, b: Complex) {
        self.real.add_product(a.re, b.re);
        self.real.add_product(a.im, b.im);
        self.imaginary.add_product(a.im, b.re);
        self.imaginary.add_product(-a.re, b.im);
    }

    fn value(&self) -> Complex {
        Complex::new(self.real.rounded_sum(), self.imaginary.rounded_sum())
    }
}

/// Streaming selected-output covariance accumulator. One transposed complex
/// solve per observation supplies all independent and correlated factors.
pub struct NoiseAccumulator<'a, 'response> {
    response: &'a PeriodicResponse<'response>,
    adjoints: Vec<Complex>,
    pairs: Vec<CovariancePair>,
    sums: Vec<CovarianceSum>,
    output_count: usize,
    factors: usize,
    terms: usize,
    work: usize,
    limits: NoiseLimits,
}

impl<'a, 'response> NoiseAccumulator<'a, 'response> {
    /// Bind selected observation adjoints and covariance pairs.
    ///
    /// Real stationary source sectors must not overlap conjugate frequencies;
    /// overlapping sectors require an augmented phase-sensitive covariance
    /// description, not an independent-sideband noise sum.
    /// # Errors
    /// Invalid or degenerate sector, invalid outputs/pairs, exceeded limits,
    /// or failed transposed sparse solve.
    pub fn new(
        response: &'a PeriodicResponse<'response>,
        outputs: &[Vec<Term>],
        pairs: &[CovariancePair],
        limits: NoiseLimits,
    ) -> SpiceResult<Self> {
        if response.plan().conjugate_overlap() {
            return Err(failure(
                "overlapping conjugate noise sectors require an augmented covariance",
            ));
        }
        if outputs.is_empty()
            || outputs.len() > limits.outputs
            || pairs.len() > limits.pairs
            || pairs
                .iter()
                .any(|pair| pair.left >= outputs.len() || pair.right >= outputs.len())
        {
            return Err(failure("invalid selected periodic noise outputs"));
        }
        let terms = outputs
            .iter()
            .try_fold(0_usize, |count, terms| count.checked_add(terms.len()))
            .filter(|&count| count <= limits.terms)
            .ok_or_else(|| failure("noise observation term budget exceeded"))?;
        let adjoints = response.adjoints(outputs, limits.adjoint_entries)?;
        let mut sums = Vec::new();
        sums.try_reserve_exact(pairs.len())
            .map_err(|_| failure("cannot allocate covariance accumulators"))?;
        sums.resize_with(pairs.len(), CovarianceSum::new);
        Ok(Self {
            response,
            adjoints,
            pairs: pairs.to_vec(),
            sums,
            output_count: outputs.len(),
            factors: 0,
            terms,
            work: 0,
            limits,
        })
    }

    fn charge_work(&mut self, work: usize) -> SpiceResult<()> {
        self.work = self
            .work
            .checked_add(work)
            .filter(|&n| n <= self.limits.work)
            .ok_or_else(|| failure("periodic noise work budget exhausted"))?;
        Ok(())
    }

    fn add_amplitudes(&mut self, amplitudes: &[Complex]) -> SpiceResult<()> {
        if amplitudes.len() != self.output_count || amplitudes.iter().any(|v| !v.is_finite()) {
            return Err(failure("invalid translated generator amplitudes"));
        }
        let factors = self
            .factors
            .checked_add(1)
            .filter(|&n| n <= self.limits.factors)
            .ok_or_else(|| failure("periodic noise factor budget exhausted"))?;
        self.charge_work(self.pairs.len())?;
        for (sum, pair) in self.sums.iter_mut().zip(&self.pairs) {
            sum.add(amplitudes[pair.left], amplitudes[pair.right]);
        }
        self.factors = factors;
        Ok(())
    }

    /// Add one independent, unit-density stationary source. Its terms are
    /// coherent complex amplitude injections across rows and sidebands,
    /// already scaled by the square root of the physical spectral density.
    /// Calling this twice adds independent sources, not coherent amplitudes.
    /// # Errors
    /// Invalid source or exceeded cumulative resource budget.
    pub fn add_factor(&mut self, terms: &[Term]) -> SpiceResult<()> {
        let count = self
            .terms
            .checked_add(terms.len())
            .filter(|&n| n <= self.limits.terms)
            .ok_or_else(|| failure("periodic source term budget exhausted"))?;
        let work = product(
            terms.len(),
            self.output_count,
            self.limits.work,
            "periodic source work budget exceeded",
        )?;
        let terms = self.response.bind_terms(terms)?;
        self.charge_work(work)?;
        let amplitudes = self
            .adjoints
            .chunks_exact(self.response.unknowns())
            .map(|adjoint| {
                terms
                    .iter()
                    .fold(Complex::ZERO, |sum, &(i, value)| sum + adjoint[i] * value)
            })
            .collect::<Vec<_>>();
        self.add_amplitudes(&amplitudes)?;
        self.terms = count;
        Ok(())
    }

    /// Add a matched external source's thermal incident wave at one port
    /// and signed input sideband. Native port reference resistances are
    /// noiseless in the circuit device description; source noise is explicit.
    /// # Errors
    /// Invalid absolute temperature, port, sideband or resource budget.
    pub fn add_thermal_input(&mut self, channel: PortChannel, kelvin: Real) -> SpiceResult<()> {
        if !kelvin.is_finite() || kelvin <= 0. {
            return Err(failure("invalid input noise temperature"));
        }
        self.add_factor(
            &self
                .response
                .incident_wave(channel, Complex::real((BOLTZMANN * kelvin).sqrt()))?,
        )
    }

    /// Finish the selected covariance, rounding each exact product sum once.
    /// # Errors
    /// An unrepresentable output density.
    pub fn finish(self) -> SpiceResult<NoiseResult> {
        let densities = self
            .sums
            .iter()
            .map(CovarianceSum::value)
            .collect::<Vec<_>>();
        if densities.iter().any(|v| !v.is_finite()) {
            return Err(failure("nonfinite translated noise covariance"));
        }
        Ok(NoiseResult {
            pairs: self.pairs,
            densities,
            factors: self.factors,
        })
    }
}

/// Gain and noise figure for a chosen input/output power-wave channel pair.
#[derive(Debug, Clone, Copy)]
pub struct NoiseFigure {
    /// Conversion power gain, |S|².
    pub gain: Real,
    /// Added output noise referred to the selected input (W/Hz).
    pub input_referred: Real,
    /// Linear noise factor.
    pub factor: Real,
    /// Noise figure in decibels.
    pub decibels: Real,
}

impl NoiseFigure {
    /// Evaluate the selected-channel noise figure. Added output noise
    /// excludes the reference input's own kT contribution; noise from
    /// other terminations and translated sidebands belongs in added noise
    /// when those sources are present.
    /// # Errors
    /// Invalid temperature, zero/unrepresentable gain or invalid density.
    pub fn from_added_noise(
        conversion: Complex,
        added_density: Real,
        reference_kelvin: Real,
    ) -> SpiceResult<Self> {
        let gain = conversion.magnitude().powi(2);
        if !gain.is_finite()
            || gain <= 0.
            || !added_density.is_finite()
            || added_density < 0.
            || !reference_kelvin.is_finite()
            || reference_kelvin <= 0.
        {
            return Err(failure("invalid conversion noise-figure inputs"));
        }
        let input_referred = added_density / gain;
        let ratio = input_referred / (BOLTZMANN * reference_kelvin);
        let factor = 1. + ratio;
        let decibels = 10. / std::f64::consts::LN_10 * ratio.ln_1p();
        if !input_referred.is_finite() || !factor.is_finite() || !decibels.is_finite() {
            return Err(failure("unrepresentable conversion noise figure"));
        }
        Ok(Self {
            gain,
            input_referred,
            factor,
            decibels,
        })
    }
}

struct PhasePoint {
    bias: Vector,
    state: StateSnapshot,
}

fn phase_points(
    response: &PeriodicResponse<'_>,
    run: &PreparedRun<'_>,
    context: &ModelContext,
) -> SpiceResult<Vec<PhasePoint>> {
    let circuit = run.circuit();
    let orbit = response.circuit_orbit()?;
    let lattice = response.lattice();
    let width = lattice.coordinates();
    let n = circuit.unknown_count();
    let history = circuit.independent_state_history()?;
    let mut spare = None;
    let mut matrix = circuit.new_matrix()?;
    let mut rhs = Vector::zeros(n);
    let mut points = Vec::new();
    points
        .try_reserve_exact(lattice.samples())
        .map_err(|_| failure("cannot allocate noise phase states"))?;
    for basis in lattice.rows() {
        let bias = Vector::from(
            orbit
                .coefficients
                .as_slice()
                .chunks_exact(width)
                .map(|row| {
                    row.iter()
                        .zip(basis)
                        .fold(0., |sum, (&a, &b)| a.mul_add(b, sum))
                })
                .collect::<Vec<_>>(),
        );
        if !bias.is_finite() {
            return Err(failure("nonfinite periodic noise orbit state"));
        }
        let mut trial = history.trial_reusing(spare.take());
        matrix.clear();
        rhs.clear();
        run.load(
            &LoadRequest {
                mode: AnalysisMode::OperatingPoint,
                solution: &bias,
                model_context: context,
                integration: None,
                history: &history,
                forcing: None,
            },
            &mut matrix,
            &mut rhs,
            &mut trial,
        )?;
        let state = run.independent_small_signal_snapshot(context, &bias, trial.values())?;
        spare = Some(trial);
        points.push(PhasePoint { bias, state });
    }
    Ok(points)
}

/// Weights exp(-j k dot theta) in signed sideband order.
fn phase_weights(basis: &[Real], weights: &mut [Complex]) {
    let center = weights.len() / 2;
    weights[center] = Complex::real(1.);
    for (i, pair) in basis[1..].chunks_exact(2).enumerate() {
        let positive = Complex::new(pair[0], -pair[1]);
        weights[center + 1 + i] = positive;
        weights[center - 1 - i] = positive.conj();
    }
}

struct NoisePhases {
    weights: Vec<Complex>,
    adjoints: Vec<Complex>,
    phases: usize,
    rows: usize,
    outputs: usize,
    bands: usize,
    storage: usize,
}

impl NoisePhases {
    fn prepare(accumulator: &mut NoiseAccumulator<'_, '_>) -> SpiceResult<Self> {
        let response = accumulator.response;
        let lattice = response.lattice();
        let phases = lattice.samples();
        let bands = lattice.coordinates();
        let rows = response.physical_unknowns();
        let outputs = accumulator.output_count;
        let max = accumulator.limits.phase_entries;
        let weight_entries = product(phases, bands, max, "periodic phase weight budget exceeded")?;
        let per_phase = product(outputs, rows, max, "periodic adjoint phase budget exceeded")?;
        let adjoint_entries = product(
            phases,
            per_phase,
            max,
            "periodic adjoint phase budget exceeded",
        )?;
        let circuit = response.circuit_orbit()?.circuit;
        let state_entries = product(
            phases,
            rows.checked_add(circuit.state_len())
                .ok_or_else(|| failure("periodic phase state size overflow"))?,
            max,
            "periodic phase state budget exceeded",
        )?;
        let storage = weight_entries
            .checked_add(adjoint_entries)
            .and_then(|n| n.checked_add(state_entries))
            .filter(|&n| n <= max)
            .ok_or_else(|| failure("aggregate periodic noise phase storage exceeded"))?;
        let work = product(
            adjoint_entries,
            bands,
            accumulator.limits.work,
            "periodic adjoint synthesis budget exceeded",
        )?;
        let state_work = product(
            rows,
            weight_entries,
            accumulator.limits.work,
            "periodic orbit synthesis budget exceeded",
        )?;
        accumulator.charge_work(
            work.checked_add(state_work)
                .ok_or_else(|| failure("periodic phase work overflow"))?,
        )?;
        let mut weights = filled_vec(
            weight_entries,
            Complex::ZERO,
            "periodic noise phase weights",
        )?;
        let mut adjoints = filled_vec(
            adjoint_entries,
            Complex::ZERO,
            "periodic noise phase adjoints",
        )?;
        for ((basis, weights), phase) in lattice
            .rows()
            .zip(weights.chunks_exact_mut(bands))
            .zip(adjoints.chunks_exact_mut(per_phase))
        {
            phase_weights(basis, weights);
            for (out, lane) in phase
                .chunks_exact_mut(rows)
                .zip(accumulator.adjoints.chunks_exact(response.unknowns()))
            {
                for (value, coefficients) in out.iter_mut().zip(lane.chunks_exact(bands)) {
                    *value = coefficients
                        .iter()
                        .zip(weights.iter())
                        .fold(Complex::ZERO, |sum, (&a, &b)| sum + a * b);
                    if !value.is_finite() {
                        return Err(failure("nonfinite periodic adjoint synthesis"));
                    }
                }
            }
        }
        Ok(Self {
            weights,
            adjoints,
            phases,
            rows,
            outputs,
            bands,
            storage,
        })
    }

    fn source_work(&self, sources: usize, limits: NoiseLimits) -> SpiceResult<usize> {
        let entries = product(
            self.phases,
            self.bands,
            limits.evaluations,
            "periodic phase/frequency evaluation budget exceeded",
        )?;
        let work = product(
            entries,
            sources,
            limits.work,
            "periodic noise source work exceeded",
        )?;
        let work = product(
            work,
            self.outputs,
            limits.work,
            "periodic noise propagation work exceeded",
        )?;
        product(
            work,
            4,
            limits.work,
            "periodic noise propagation work exceeded",
        )
    }

    fn source_phase(&self, phase: usize, band: usize) -> (Complex, &[Complex]) {
        let per_phase = self.rows * self.outputs;
        (
            self.weights[phase * self.bands + band].conj(),
            &self.adjoints[phase * per_phase..(phase + 1) * per_phase],
        )
    }
}

// A compensated coherent Fourier sum; covariance itself uses exact products.
#[derive(Clone, Copy)]
struct CoherentSum {
    sum: Complex,
    correction: Complex,
}

impl CoherentSum {
    const ZERO: Self = Self {
        sum: Complex::ZERO,
        correction: Complex::ZERO,
    };

    fn add(&mut self, value: Complex) {
        let corrected = value - self.correction;
        let next = self.sum + corrected;
        self.correction = (next - self.sum) - corrected;
        self.sum = next;
    }
}

fn add_native_band<'a>(
    plan: &mut NoisePlan<'a>,
    run: &PreparedRun<'a>,
    points: &[PhasePoint],
    context: &ModelContext,
    frequency: Real,
    band: usize,
    phases: &NoisePhases,
    sums: &mut [CoherentSum],
) -> SpiceResult<()> {
    sums.fill(CoherentSum::ZERO);
    let context = context.with_frequency(frequency.abs());
    let normalizer = 1.
        / Real::from(
            u32::try_from(points.len()).map_err(|_| failure("noise phase count overflow"))?,
        );
    for (phase, point) in points.iter().enumerate() {
        plan.refresh(run, &context, &point.bias, Some(&point.state))?;
        let (weight, adjoints) = phases.source_phase(phase, band);
        let weight = weight * Complex::real(normalizer);
        let generators = plan.instances().flat_map(|(_, sources)| sources);
        for (source, outputs) in generators.zip(sums.chunks_exact_mut(phases.outputs)) {
            let terms = source.periodic_amplitudes(frequency)?;
            for (sum, adjoint) in outputs.iter_mut().zip(adjoints.chunks_exact(phases.rows)) {
                let value = terms
                    .into_iter()
                    .fold(Complex::ZERO, |sum, (row, amplitude)| {
                        row.map_or(sum, |row| sum + adjoint[row] * amplitude)
                    });
                sum.add(value * weight);
            }
        }
    }
    Ok(())
}

impl NoiseAccumulator<'_, '_> {
    /// Add the circuit's native thermal, shot, flicker and coherent noise
    /// generators, evaluated at every physical phase and signed source
    /// sideband. Each named generator is a stationary independent process
    /// multiplied by its phase-local complex spectral amplitude. Sources
    /// with additional temporal memory need an explicit external factor
    /// model, not a DC-linearized substitute.
    /// # Errors
    /// No bound circuit, unsupported/changed generator layout, invalid
    /// spectral density, exceeded budgets or failed physical phase load.
    pub fn add_circuit_noise(&mut self) -> SpiceResult<()> {
        let previous = self.sums.clone();
        let (factors, work) = (self.factors, self.work);
        if let Err(error) = self.add_native_noise() {
            self.sums = previous;
            self.factors = factors;
            self.work = work;
            return Err(error);
        }
        Ok(())
    }

    fn add_native_noise(&mut self) -> SpiceResult<()> {
        let orbit = self.response.circuit_orbit()?;
        let run = orbit.circuit.prepare_run();
        let lattice = self.response.lattice();
        product(
            lattice.samples(),
            lattice.coordinates() + 1,
            self.limits.evaluations.saturating_sub(1),
            "periodic phase/frequency evaluation budget exceeded",
        )?;
        let phases = NoisePhases::prepare(self)?;
        let points = phase_points(self.response, &run, &orbit.context)?;
        let first = self.response.plan().bands()[0].frequency().abs();
        let mut plan = NoisePlan::bind(
            &run,
            &orbit.context.with_frequency(first),
            &points[0].bias,
            Some(&points[0].state),
        )?;
        let count = plan.source_count();
        let factors = product(
            count,
            phases.bands,
            self.limits.factors,
            "periodic native factor budget exceeded",
        )?;
        if self
            .factors
            .checked_add(factors)
            .is_none_or(|n| n > self.limits.factors)
        {
            return Err(failure("periodic native factor budget exceeded"));
        }
        let work = phases.source_work(count, self.limits)?;
        let covariance_work = product(
            factors,
            self.pairs.len(),
            self.limits.work,
            "periodic covariance work budget exceeded",
        )?;
        if self
            .work
            .checked_add(work)
            .and_then(|n| n.checked_add(covariance_work))
            .is_none_or(|n| n > self.limits.work)
        {
            return Err(failure("aggregate periodic native noise work exceeded"));
        }
        self.charge_work(work)?;
        let entries = product(
            count,
            self.output_count,
            self.limits.phase_entries,
            "periodic generator workspace exceeded",
        )?;
        if entries
            .checked_mul(2)
            .and_then(|n| n.checked_add(phases.storage))
            .and_then(|n| n.checked_add(self.output_count))
            .is_none_or(|n| n > self.limits.phase_entries)
        {
            return Err(failure(
                "aggregate periodic noise generator storage exceeded",
            ));
        }
        let mut sums = filled_vec(
            entries,
            CoherentSum::ZERO,
            "periodic coherent generator workspace",
        )?;
        let mut amplitudes = filled_vec(
            self.output_count,
            Complex::ZERO,
            "periodic generator output amplitudes",
        )?;
        for (band, coordinate) in self.response.plan().bands().iter().enumerate() {
            add_native_band(
                &mut plan,
                &run,
                &points,
                &orbit.context,
                coordinate.frequency(),
                band,
                &phases,
                &mut sums,
            )?;
            for source in sums.chunks_exact(self.output_count) {
                for (amplitude, sum) in amplitudes.iter_mut().zip(source) {
                    *amplitude = sum.sum;
                }
                self.add_amplitudes(&amplitudes)?;
            }
        }
        Ok(())
    }
}
