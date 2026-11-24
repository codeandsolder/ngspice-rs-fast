//! Signed Floquet sideband coordinates for perturbations of a periodic orbit.

use super::harmonic_balance::HarmonicLattice;
use crate::primitives::{Real, SpiceError, SpiceResult};
use std::{
    f64::consts::TAU,
    sync::{Arc, Mutex},
};

fn failure(message: &str) -> SpiceError {
    SpiceError::Numerical {
        context: "periodic small signal".into(),
        message: message.into(),
    }
}

/// One complex perturbation coordinate, relative to an independent probe.
#[derive(Debug, Clone, PartialEq)]
pub struct Sideband {
    indices: Vec<i32>,
    frequency: Real,
}

impl Sideband {
    /// Integer offsets along the pump axes in lattice order.
    #[must_use]
    pub fn indices(&self) -> &[i32] {
        &self.indices
    }

    /// Signed physical frequency, in hertz.
    #[must_use]
    pub const fn frequency(&self) -> Real {
        self.frequency
    }
}

/// A single complex Floquet sector in ascending signed physical frequency.
///
/// The complex envelope is not folded onto positive frequencies: a real
/// perturbation also has the conjugate sector at the negative probe.
#[derive(Debug, Clone, PartialEq)]
pub struct SidebandPlan {
    probe: Real,
    bands: Vec<Sideband>,
    conjugate_overlap: bool,
}

fn coincides(a: Real, b: Real) -> bool {
    (a - b).abs() <= 8. * Real::EPSILON * a.abs().max(b.abs())
}

impl SidebandPlan {
    /// Derive signed sideband coordinates from the retained pump lattice.
    ///
    /// # Errors
    /// Invalid probe frequency, exceeded storage budget, unrepresentable
    /// shifted frequencies or collapsed distinct harmonic coordinates.
    pub fn new(lattice: &HarmonicLattice, probe: Real, max_sidebands: usize) -> SpiceResult<Self> {
        if !probe.is_finite() || lattice.coordinates() > max_sidebands {
            return Err(failure("invalid probe or sideband budget exceeded"));
        }
        let mut bands = Vec::new();
        bands
            .try_reserve_exact(lattice.coordinates())
            .map_err(|_| failure("cannot allocate sideband plan"))?;
        for harmonic in lattice.harmonics().iter().rev() {
            bands.push(Sideband {
                indices: harmonic.indices().iter().map(|&index| -index).collect(),
                frequency: probe - harmonic.frequency(),
            });
        }
        bands.push(Sideband {
            indices: vec![0; lattice.axes().len()],
            frequency: probe,
        });
        for harmonic in lattice.harmonics() {
            bands.push(Sideband {
                indices: harmonic.indices().to_vec(),
                frequency: probe + harmonic.frequency(),
            });
        }
        if bands.iter().any(|band| !(TAU * band.frequency).is_finite())
            || bands
                .windows(2)
                .any(|pair| coincides(pair[0].frequency, pair[1].frequency))
        {
            return Err(failure("nonfinite or indistinguishable shifted sidebands"));
        }
        // The reflected sector is already sorted by reversing the original
        // frequencies and changing their sign. A linear merge detects
        // phase-sensitive real-signal overlap without a quadratic search.
        let mut left = 0;
        let mut right = bands.len();
        let mut conjugate_overlap = false;
        while left < bands.len() && right > 0 {
            let a = bands[left].frequency;
            let b = -bands[right - 1].frequency;
            if coincides(a, b) {
                conjugate_overlap = true;
                break;
            }
            if a < b {
                left += 1;
            } else {
                right -= 1;
            }
        }
        Ok(Self {
            probe,
            bands,
            conjugate_overlap,
        })
    }

    /// Signed probe frequency, before pump offsets.
    #[must_use]
    pub const fn probe(&self) -> Real {
        self.probe
    }

    /// Ascending signed sideband coordinates.
    #[must_use]
    pub fn bands(&self) -> &[Sideband] {
        &self.bands
    }

    /// Whether the opposite-probe sector contains an overlapping frequency.
    /// Real-signal power or noise cannot be counted as independent sectors
    /// in this case; their coherent amplitudes must first be combined.
    #[must_use]
    pub const fn conjugate_overlap(&self) -> bool {
        self.conjugate_overlap
    }

    /// Combine one signed complex sector into a physical real-signal peak
    /// phasor at a nonnegative frequency. A negative-frequency contribution
    /// enters conjugated; overlapping sectors are added coherently. At DC
    /// only the real part survives.
    /// # Errors
    /// Invalid shape, nonfinite amplitudes or absent target frequency.
    pub fn real_phasor(&self, amplitudes: &[Complex], frequency: Real) -> SpiceResult<Complex> {
        if amplitudes.len() != self.bands.len()
            || !frequency.is_finite()
            || frequency < 0.
            || amplitudes.iter().any(|value| !value.is_finite())
        {
            return Err(failure("invalid real-signal sideband observation"));
        }
        let mut result = Complex::ZERO;
        let mut present = false;
        for (band, &amplitude) in self.bands.iter().zip(amplitudes) {
            if coincides(band.frequency, frequency) {
                result = result + amplitude;
                present = true;
            } else if coincides(band.frequency, -frequency) {
                result = result + amplitude.conj();
                present = true;
            }
        }
        if frequency == 0. {
            result.im = 0.;
        }
        if !present || !result.is_finite() {
            return Err(failure("absent or nonfinite real-signal observation"));
        }
        Ok(result)
    }

    /// Locate the exact integer pump offset (not an approximate Hz key).
    #[must_use]
    pub fn locate(&self, indices: &[i32]) -> Option<usize> {
        self.bands.iter().position(|band| band.indices == indices)
    }
}

use super::harmonic_balance::{
    BalanceLimits, BalanceSolution, HarmonicModel, OrbitSpectrum, linearize_circuit_orbit,
    linearize_orbit,
};
use crate::devices::{Circuit, ModelContext};
use crate::maths::complex::{ComplexLu, ComplexMatrix};
use crate::maths::equilibration::OriginalResidual;
use crate::maths::linear::{RHS_BATCH_WIDTH, SolveBatchWorkspace};
use crate::primitives::{Complex, filled_vec};

/// Frequency-local excitation or observation coordinate, bound by physical
/// MNA row and signed sideband ordinal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Coordinate {
    /// Physical MNA equation/unknown row.
    pub row: usize,
    /// Ordinal in the signed sideband plan.
    pub band: usize,
}

/// A sparse linear combination of physical sideband coordinates.
#[derive(Debug, Clone, Copy)]
pub struct Term {
    /// Physical coordinate.
    pub at: Coordinate,
    /// Complex coefficient (no implicit conjugation).
    pub value: Complex,
}

/// Limits for compact response queries. The HB budget separately bounds
/// physical sampling, expanded matrix storage and worst-case factor fill.
#[derive(Debug, Clone, Copy)]
pub struct ResponseLimits {
    /// Maximum number of input/output transfer coefficients returned.
    pub outputs: usize,
    /// Maximum number of weighted terms across requested inputs and outputs.
    pub terms: usize,
    /// Maximum scalar cells in each RHS/solution tile.
    pub tile_entries: usize,
    /// Optional maximum componentwise backward error against the original
    /// complex operator, checked with exact residual accumulation.
    pub max_backward_error: Option<Real>,
}

impl Default for ResponseLimits {
    fn default() -> Self {
        Self {
            outputs: 1_000_000,
            terms: 1_000_000,
            tile_entries: 1_000_000,
            max_backward_error: None,
        }
    }
}

/// A bound real-reference RF port, using the native Thevenin branch current.
#[derive(Debug, Clone, Copy)]
pub struct PeriodicPort {
    /// One-based circuit port number.
    pub number: usize,
    /// Reference resistance in ohms.
    pub reference: Real,
    positive: Option<usize>,
    negative: Option<usize>,
    branch: usize,
}

/// One signed RF channel.
#[derive(Debug, Clone, Copy)]
pub struct PortChannel {
    /// One-based port number.
    pub port: usize,
    /// Signed sideband ordinal.
    pub band: usize,
}

/// Revalidated periodic orbit and projected physical derivatives.
///
/// The lattice and immutable derivative spectra are reused across probe
/// frequencies. Every factor is an independent shifted complex operator.
pub struct PeriodicLinearization<'a> {
    lattice: &'a HarmonicLattice,
    spectrum: OrbitSpectrum,
    ports: Vec<PeriodicPort>,
    last_factor: Mutex<Option<Arc<FactorSnapshot>>>,
    circuit_orbit: Option<CircuitOrbit<'a>>,
    pivot: crate::maths::PivotTolerances,
}

/// The physical noise model is bound to the same immutable circuit,
/// candidate context and revalidated orbit as the response operator.
pub(super) struct CircuitOrbit<'a> {
    pub(super) circuit: &'a Circuit,
    pub(super) context: ModelContext,
    pub(super) coefficients: crate::maths::Vector,
}

impl<'a> PeriodicLinearization<'a> {
    /// Prepare a physical model from a supplied HB solution.
    /// # Errors
    /// Invalid or unconverged reference, exceeded resource limits, or
    /// unavailable physical current/charge derivatives.
    pub fn prepare(
        lattice: &'a HarmonicLattice,
        model: &mut impl HarmonicModel,
        orbit: &BalanceSolution,
        tolerance: &[Real],
        limits: BalanceLimits,
    ) -> SpiceResult<Self> {
        Ok(Self {
            lattice,
            spectrum: linearize_orbit(lattice, model, orbit, tolerance, limits)?,
            ports: Vec::new(),
            last_factor: Mutex::default(),
            circuit_orbit: None,
            pivot: crate::maths::PivotTolerances::DEFAULT,
        })
    }

    /// Prepare a finalized circuit at the same physical candidate context
    /// used for its HB orbit, binding its RF port branches once.
    /// # Errors
    /// Invalid model, unconverged reference, RF port numbering or resource limits.
    pub fn circuit(
        lattice: &'a HarmonicLattice,
        circuit: &'a mut Circuit,
        context: ModelContext,
        orbit: &BalanceSolution,
        tolerance: &[Real],
        limits: BalanceLimits,
    ) -> SpiceResult<Self> {
        circuit.finalize()?;
        let spectrum =
            linearize_circuit_orbit(circuit, context, lattice, orbit, tolerance, limits)?;
        let mut ports = Vec::new();
        for (index, device) in circuit.devices().iter().enumerate() {
            if let Some(port) = device.rf_port() {
                let branch = circuit
                    .branch_rows(index)
                    .filter(|rows| rows.len() == 1)
                    .ok_or_else(|| failure("invalid periodic RF port branch"))?
                    .start;
                let row = |i: usize| circuit.unknowns().checked_node_row(device.terminals()[i]);
                ports.push(PeriodicPort {
                    number: port.number,
                    reference: port.z0,
                    positive: row(0)?,
                    negative: row(1)?,
                    branch,
                });
            }
        }
        ports.sort_unstable_by_key(|port| port.number);
        if ports
            .iter()
            .enumerate()
            .any(|(index, port)| port.number != index + 1)
        {
            return Err(failure("periodic RF ports must be numbered 1..=N"));
        }
        Ok(Self {
            lattice,
            spectrum,
            ports,
            last_factor: Mutex::default(),
            pivot: context.pivot,
            circuit_orbit: Some(CircuitOrbit {
                circuit,
                context,
                coefficients: orbit.coefficients.clone(),
            }),
        })
    }

    /// Rechecked maximum normalized physical HB residual.
    #[must_use]
    pub const fn scaled_residual(&self) -> Real {
        self.spectrum.scaled_residual
    }

    /// Immutable native RF port bindings.
    #[must_use]
    pub fn ports(&self) -> &[PeriodicPort] {
        &self.ports
    }

    /// Assemble and factor one Floquet sector.
    ///
    /// Blocks are G[k-l] + j 2 pi f[k] C[k-l], where the derivative is
    /// taken at the output frequency, not at the input/probe frequency.
    /// # Errors
    /// Collapsed sidebands, nonfinite coefficients or singular operator.
    pub fn factor(&self, probe: Real) -> SpiceResult<PeriodicResponse<'_>> {
        if let Some(snapshot) = self
            .last_factor
            .lock()
            .map_err(|_| failure("periodic factor cache poisoned"))?
            .as_ref()
            && snapshot.plan.probe().to_bits() == probe.to_bits()
        {
            return Ok(PeriodicResponse {
                linearization: self,
                snapshot: Arc::clone(snapshot),
            });
        }
        let spectrum = &self.spectrum;
        let plan = SidebandPlan::new(self.lattice, probe, spectrum.width)?;
        let mut values = filled_vec(
            spectrum.pattern.len(),
            Complex::ZERO,
            "periodic complex operator",
        )?;
        let frequencies: Vec<_> = plan
            .bands()
            .iter()
            .map(|band| TAU * band.frequency())
            .collect();
        for ((value, (row, _)), &index) in values
            .iter_mut()
            .zip(spectrum.pattern.positions())
            .zip(&spectrum.spectral_indices)
        {
            *value = spectrum.conductance[index]
                + Complex::imaginary(frequencies[row % spectrum.width])
                    * spectrum.capacitance[index];
        }
        let matrix =
            ComplexMatrix::from_bound_values(spectrum.pattern.clone(), values, self.pivot)?;
        let factor = matrix.factorize()?;
        let snapshot = Arc::new(FactorSnapshot {
            plan,
            matrix,
            factor,
        });
        *self
            .last_factor
            .lock()
            .map_err(|_| failure("periodic factor cache poisoned"))? = Some(Arc::clone(&snapshot));
        Ok(PeriodicResponse {
            linearization: self,
            snapshot,
        })
    }
}

struct FactorSnapshot {
    plan: SidebandPlan,
    matrix: ComplexMatrix,
    factor: ComplexLu,
}

/// Factored Floquet response at a single probe. All selected excitations
/// and translated-noise queries share the same immutable numeric factors.
pub struct PeriodicResponse<'a> {
    linearization: &'a PeriodicLinearization<'a>,
    snapshot: Arc<FactorSnapshot>,
}

impl PeriodicResponse<'_> {
    /// Signed frequencies and integer offsets of this response.
    #[must_use]
    pub fn plan(&self) -> &SidebandPlan {
        &self.snapshot.plan
    }

    /// Actual sparse numeric factor storage, including any fill.
    #[must_use]
    pub fn factor_storage(&self) -> crate::maths::FactorStorage {
        self.snapshot.factor.factor_storage()
    }

    /// True when both responses share the same immutable numeric factors.
    #[must_use]
    pub fn shares_factor(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.snapshot, &other.snapshot)
    }

    /// Physical matrix order before expanding sidebands.
    #[must_use]
    pub const fn physical_unknowns(&self) -> usize {
        self.linearization.spectrum.unknowns
    }

    /// Expanded complex matrix order.
    #[must_use]
    pub const fn unknowns(&self) -> usize {
        self.linearization.spectrum.unknowns * self.linearization.spectrum.width
    }

    pub(super) const fn lattice(&self) -> &HarmonicLattice {
        self.linearization.lattice
    }

    pub(super) fn circuit_orbit(&self) -> SpiceResult<&CircuitOrbit<'_>> {
        self.linearization
            .circuit_orbit
            .as_ref()
            .ok_or_else(|| failure("periodic response has no bound circuit noise model"))
    }

    fn index(&self, at: Coordinate) -> SpiceResult<usize> {
        if at.row >= self.physical_unknowns() || at.band >= self.plan().bands().len() {
            return Err(failure("periodic coordinate out of bounds"));
        }
        Ok(at.row * self.plan().bands().len() + at.band)
    }

    pub(crate) fn bind_terms(&self, terms: &[Term]) -> SpiceResult<Vec<(usize, Complex)>> {
        terms
            .iter()
            .map(|term| {
                if !term.value.is_finite() {
                    return Err(failure("nonfinite periodic excitation or observation"));
                }
                Ok((self.index(term.at)?, term.value))
            })
            .collect()
    }

    /// Solve arbitrary selected complex responses in RHS-major order.
    /// Each input is a sparse equation RHS; each output is a sparse linear
    /// combination of unknowns. No full time/sideband plots are retained.
    /// # Errors
    /// Invalid coordinates, exceeded query limits, or failed sparse solve.
    pub fn transfer(
        &self,
        inputs: &[Vec<Term>],
        outputs: &[Vec<Term>],
        limits: ResponseLimits,
    ) -> SpiceResult<Vec<Complex>> {
        let count = inputs
            .len()
            .checked_mul(outputs.len())
            .filter(|&n| n <= limits.outputs)
            .ok_or_else(|| failure("periodic response output budget exceeded"))?;
        inputs
            .iter()
            .chain(outputs)
            .try_fold(0_usize, |n, terms| n.checked_add(terms.len()))
            .filter(|&n| n <= limits.terms)
            .ok_or_else(|| failure("periodic response term budget exceeded"))?;
        if inputs.is_empty() || outputs.is_empty() {
            return Ok(Vec::new());
        }
        let n = self.unknowns();
        let width = RHS_BATCH_WIDTH.min(limits.tile_entries / n);
        if width == 0 {
            return Err(failure("periodic RHS tile budget exceeded"));
        }
        let observed = outputs
            .iter()
            .map(|terms| self.bind_terms(terms))
            .collect::<SpiceResult<Vec<_>>>()?;
        let mut result = Vec::new();
        result
            .try_reserve_exact(count)
            .map_err(|_| failure("cannot allocate periodic response"))?;
        let mut rhs = filled_vec(width * n, Complex::ZERO, "periodic RHS tile")?;
        let mut solution = Vec::new();
        let mut scratch = SolveBatchWorkspace::default();
        let entries = if let Some(bound) = limits.max_backward_error {
            if !bound.is_finite() || bound <= 0. {
                return Err(failure("invalid periodic backward-error tolerance"));
            }
            let (pattern, values) = self
                .snapshot
                .matrix
                .bound_coefficients()
                .ok_or_else(|| failure("missing periodic original operator"))?;
            Some(
                pattern
                    .positions()
                    .zip(values)
                    .map(|((r, c), &v)| (r, c, v))
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        };
        let residual = entries
            .as_ref()
            .map(|entries| OriginalResidual::complex(n, entries, false))
            .transpose()?;
        for tile in inputs.chunks(width) {
            let rhs = &mut rhs[..tile.len() * n];
            rhs.fill(Complex::ZERO);
            for (lane, input) in rhs.chunks_exact_mut(n).zip(tile) {
                for (index, value) in self.bind_terms(input)? {
                    lane[index] = lane[index] + value;
                }
            }
            self.snapshot
                .factor
                .solve_batch_into(rhs, &mut scratch, &mut solution, false)?;
            for (lane, source) in solution.chunks_exact(n).zip(rhs.chunks_exact(n)) {
                if let (Some(residual), Some(bound)) = (&residual, limits.max_backward_error) {
                    let error = residual.evaluate(source, lane)?.maximum;
                    if error > bound {
                        return Err(failure("periodic complex backward error exceeds tolerance"));
                    }
                }
                for terms in &observed {
                    let value = terms.iter().fold(Complex::ZERO, |sum, &(index, weight)| {
                        sum + lane[index] * weight
                    });
                    if !value.is_finite() {
                        return Err(failure("nonfinite periodic observation"));
                    }
                    result.push(value);
                }
            }
        }
        Ok(result)
    }

    pub(crate) fn adjoints(
        &self,
        outputs: &[Vec<Term>],
        max_entries: usize,
    ) -> SpiceResult<Vec<Complex>> {
        let n = self.unknowns();
        let count = n
            .checked_mul(outputs.len())
            .filter(|&count| count <= max_entries)
            .ok_or_else(|| failure("periodic noise adjoint budget exceeded"))?;
        if outputs.is_empty() {
            return Ok(Vec::new());
        }
        let mut result = Vec::new();
        result
            .try_reserve_exact(count)
            .map_err(|_| failure("cannot allocate periodic adjoints"))?;
        let mut rhs = filled_vec(
            n * RHS_BATCH_WIDTH.min(outputs.len()),
            Complex::ZERO,
            "periodic adjoint RHS",
        )?;
        let mut scratch = SolveBatchWorkspace::default();
        let mut solved = Vec::new();
        for tile in outputs.chunks(RHS_BATCH_WIDTH) {
            let rhs = &mut rhs[..tile.len() * n];
            rhs.fill(Complex::ZERO);
            for (lane, output) in rhs.chunks_exact_mut(n).zip(tile) {
                for (index, value) in self.bind_terms(output)? {
                    lane[index] = lane[index] + value;
                }
            }
            self.snapshot
                .factor
                .solve_batch_into(rhs, &mut scratch, &mut solved, true)?;
            result.extend_from_slice(&solved);
        }
        Ok(result)
    }

    fn channel(&self, channel: PortChannel) -> SpiceResult<PeriodicPort> {
        if channel.band >= self.plan().bands().len() {
            return Err(failure("invalid RF channel sideband"));
        }
        channel
            .port
            .checked_sub(1)
            .and_then(|index| self.linearization.ports.get(index))
            .copied()
            .ok_or_else(|| failure("invalid RF channel port"))
    }

    /// A native port's incident-wave excitation normalized to the given
    /// complex amplitude in power-wave units.
    /// # Errors
    /// Invalid port, sideband or amplitude.
    pub fn incident_wave(
        &self,
        channel: PortChannel,
        amplitude: Complex,
    ) -> SpiceResult<Vec<Term>> {
        let port = self.channel(channel)?;
        let value = amplitude * Complex::real(2. * port.reference.sqrt());
        if !value.is_finite() {
            return Err(failure("nonfinite incident wave"));
        }
        Ok(vec![Term {
            at: Coordinate {
                row: port.branch,
                band: channel.band,
            },
            value,
        }])
    }

    /// Sparse observation of the outgoing wave on a native RF channel.
    /// # Errors
    /// Invalid port or sideband.
    pub fn scattered_wave(&self, channel: PortChannel) -> SpiceResult<Vec<Term>> {
        let port = self.channel(channel)?;
        let ki = 0.5 / port.reference.sqrt();
        let mut terms = vec![Term {
            at: Coordinate {
                row: port.branch,
                band: channel.band,
            },
            value: Complex::real(0.5 * port.reference.sqrt()),
        }];
        for (row, sign) in [(port.positive, 1.), (port.negative, -1.)] {
            if let Some(row) = row {
                terms.push(Term {
                    at: Coordinate {
                        row,
                        band: channel.band,
                    },
                    value: Complex::real(sign * ki),
                });
            }
        }
        Ok(terms)
    }

    /// Selected power-wave conversion coefficients, input-major. Native
    /// source branches are driven by 2 sqrt(Z0) for one incident wave.
    /// Negative frequencies remain complex signed channels. For a real
    /// excitation with conjugate overlap, combine coherent sectors first.
    /// # Errors
    /// Invalid channel, resource limit or failed complex solve.
    pub fn scattering(
        &self,
        inputs: &[PortChannel],
        outputs: &[PortChannel],
        limits: ResponseLimits,
    ) -> SpiceResult<Vec<Complex>> {
        if inputs
            .len()
            .checked_mul(outputs.len())
            .is_none_or(|n| n > limits.outputs)
            || inputs
                .len()
                .checked_add(outputs.len().saturating_mul(3))
                .is_none_or(|n| n > limits.terms)
        {
            return Err(failure("periodic RF channel budget exceeded"));
        }
        let excitation = inputs
            .iter()
            .map(|&channel| self.incident_wave(channel, Complex::real(1.)))
            .collect::<SpiceResult<Vec<_>>>()?;
        let observed = outputs
            .iter()
            .map(|&channel| self.scattered_wave(channel))
            .collect::<SpiceResult<Vec<_>>>()?;
        self.transfer(&excitation, &observed, limits)
    }
}
