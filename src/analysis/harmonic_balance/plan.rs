//! Real Fourier coordinates on a bounded, multidimensional tone lattice.

use std::f64::consts::TAU;

use crate::primitives::{Complex, Real, SpiceResult};

use super::failure;

/// One independent fundamental and its symmetric retained index range.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ToneAxis {
    /// Fundamental in hertz.
    pub frequency: Real,
    /// Retain indices from minus this order to plus this order.
    pub order: u16,
}

/// Maximum admitted storage and transform sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LatticeLimits {
    /// Maximum independent phase dimensions.
    pub dimensions: usize,
    /// Maximum real Fourier coordinates (including DC).
    pub coordinates: usize,
    /// Maximum collocation samples.
    pub samples: usize,
    /// Maximum scalar entries in the synthesis basis.
    pub basis_entries: usize,
}

impl Default for LatticeLimits {
    fn default() -> Self {
        Self {
            dimensions: 4,
            coordinates: 1025,
            samples: 65_536,
            basis_entries: 8_000_000,
        }
    }
}

/// One positive physical frequency and its signed combination indices.
/// The negative-frequency coefficient is its complex conjugate.
#[derive(Debug, Clone, PartialEq)]
pub struct Harmonic {
    indices: Vec<i32>,
    frequency: Real,
}

impl Harmonic {
    /// Integer coefficients of the independent fundamentals.
    #[must_use]
    pub fn indices(&self) -> &[i32] {
        &self.indices
    }

    /// Positive physical frequency in hertz.
    #[must_use]
    pub const fn frequency(&self) -> Real {
        self.frequency
    }
}

/// A real Fourier representation of a rectangular harmonic lattice.
///
/// Coordinate zero is DC; each positive-frequency mode has a cosine and
/// sine coordinate. The torus grid is independent of physical time: even
/// very close fundamentals do not require a long common-period simulation.
/// Related tones should be represented as integer multiples of the same
/// independent axis. Distinct retained combinations that collapse to one
/// physical frequency are rejected, not counted twice.
#[derive(Debug, Clone, PartialEq)]
pub struct HarmonicLattice {
    axes: Vec<ToneAxis>,
    harmonics: Vec<Harmonic>,
    samples: usize,
    basis: Vec<Real>,
}

fn checked_product(a: usize, b: usize, limit: usize, description: &str) -> SpiceResult<usize> {
    a.checked_mul(b)
        .filter(|count| *count <= limit)
        .ok_or_else(|| failure(format!("{description} budget exceeded")))
}

fn frequency(indices: &[i32], axes: &[ToneAxis]) -> (Real, Real) {
    let mut sum = 0.;
    let mut magnitude = 0.;
    for (&index, axis) in indices.iter().zip(axes) {
        sum = Real::from(index).mul_add(axis.frequency, sum);
        magnitude = Real::from(index).abs().mul_add(axis.frequency, magnitude);
    }
    (sum, magnitude)
}

fn enumerate_modes(axes: &[ToneAxis], count: usize) -> SpiceResult<Vec<Harmonic>> {
    let mut modes = Vec::new();
    modes
        .try_reserve_exact(count / 2)
        .map_err(|error| failure(error.to_string()))?;
    for ordinal in 0..count {
        let mut position = ordinal;
        let mut indices = Vec::with_capacity(axes.len());
        for axis in axes {
            let width = usize::from(axis.order) * 2 + 1;
            let digit =
                i32::try_from(position % width).map_err(|_| failure("harmonic index overflow"))?;
            indices.push(digit - i32::from(axis.order));
            position /= width;
        }
        if indices.iter().all(|index| *index == 0) {
            continue;
        }
        let (frequency, magnitude) = frequency(&indices, axes);
        if !frequency.is_finite() || !magnitude.is_finite() || !(TAU * frequency).is_finite() {
            return Err(failure("nonfinite harmonic frequency"));
        }
        if frequency.abs() <= 8. * Real::EPSILON * magnitude {
            return Err(failure(
                "dependent tone axes produce a zero-frequency harmonic; use their common fundamental",
            ));
        }
        if frequency > 0. {
            modes.push(Harmonic { indices, frequency });
        }
    }
    modes.sort_by(|a, b| a.frequency.total_cmp(&b.frequency));
    if modes.len() != count / 2
        || modes.windows(2).any(|pair| {
            let a = pair[0].frequency;
            let b = pair[1].frequency;
            b - a <= 8. * Real::EPSILON * a.max(b)
        })
    {
        return Err(failure(
            "dependent tone axes produce duplicate harmonic frequencies; use their common fundamental",
        ));
    }
    Ok(modes)
}

fn basis_rows(
    axes: &[ToneAxis],
    harmonics: &[Harmonic],
    grid: &[u32],
    samples: usize,
    entries: usize,
) -> SpiceResult<Vec<Real>> {
    let mut basis = Vec::new();
    basis
        .try_reserve_exact(entries)
        .map_err(|error| failure(error.to_string()))?;
    let mut phases = vec![0.; axes.len()];
    for ordinal in 0..samples {
        let mut position = ordinal;
        for (phase, &width) in phases.iter_mut().zip(grid) {
            let width_usize = usize::try_from(width).map_err(|_| failure("phase grid overflow"))?;
            let digit = u32::try_from(position % width_usize)
                .map_err(|_| failure("phase index overflow"))?;
            *phase = TAU * Real::from(digit) / Real::from(width);
            position /= width_usize;
        }
        basis.push(1.);
        for harmonic in harmonics {
            let angle = harmonic
                .indices
                .iter()
                .zip(&phases)
                .fold(0., |sum, (&index, &phase)| {
                    Real::from(index).mul_add(phase, sum)
                });
            let (sin, cos) = angle.sin_cos();
            basis.extend([cos, sin]);
        }
    }
    Ok(basis)
}

// Retain only checked shape metadata in analysis plans. The potentially
// large Fourier basis is allocated when the analysis executes.
#[derive(Debug, Clone, PartialEq)]
struct LatticeShape {
    axes: Vec<ToneAxis>,
    harmonics: Vec<Harmonic>,
    grid: Vec<u32>,
    samples: usize,
    entries: usize,
}

impl LatticeShape {
    fn new(axes: &[ToneAxis], oversampling: u16, limits: LatticeLimits) -> SpiceResult<Self> {
        if axes.is_empty() || axes.len() > limits.dimensions || oversampling < 2 {
            return Err(failure(
                "invalid harmonic lattice dimensions or oversampling",
            ));
        }
        let mut count = 1usize;
        let mut samples = 1usize;
        let mut grid = Vec::with_capacity(axes.len());
        for axis in axes {
            if !axis.frequency.is_finite() || axis.frequency <= 0. || axis.order == 0 {
                return Err(failure(
                    "tone frequency and harmonic order must be positive",
                ));
            }
            let width = usize::from(axis.order) * 2 + 1;
            count = checked_product(count, width, limits.coordinates, "harmonic coordinate")?;
            let points = width * usize::from(oversampling);
            samples = checked_product(samples, points, limits.samples, "collocation sample")?;
            grid.push(u32::try_from(points).map_err(|_| failure("phase grid overflow"))?);
        }
        let entries = checked_product(samples, count, limits.basis_entries, "Fourier basis")?;
        let harmonics = enumerate_modes(axes, count)?;
        Ok(Self {
            axes: axes.to_vec(),
            harmonics,
            grid,
            samples,
            entries,
        })
    }

    fn materialize(&self) -> SpiceResult<HarmonicLattice> {
        let basis = basis_rows(
            &self.axes,
            &self.harmonics,
            &self.grid,
            self.samples,
            self.entries,
        )?;
        Ok(HarmonicLattice {
            axes: self.axes.clone(),
            harmonics: self.harmonics.clone(),
            samples: self.samples,
            basis,
        })
    }
}

impl HarmonicLattice {
    /// Construct a rectangular lattice with an explicit anti-aliasing grid.
    ///
    /// Each axis uses `oversampling * (2*order + 1)` phase points, with
    /// oversampling at least two. Non-polynomial nonlinearities still require
    /// convergence checks as the grid and retained harmonic order increase.
    ///
    /// # Errors
    /// Invalid or dependent axes, invalid grid, or exceeded resource limits.
    pub fn new(axes: &[ToneAxis], oversampling: u16, limits: LatticeLimits) -> SpiceResult<Self> {
        LatticeShape::new(axes, oversampling, limits)?.materialize()
    }

    /// Independent phase axes.
    #[must_use]
    pub fn axes(&self) -> &[ToneAxis] {
        &self.axes
    }

    /// Positive-frequency harmonics in increasing physical-frequency order.
    #[must_use]
    pub fn harmonics(&self) -> &[Harmonic] {
        &self.harmonics
    }

    /// Number of real Fourier coordinates per MNA row.
    #[must_use]
    pub const fn coordinates(&self) -> usize {
        1 + self.harmonics.len() * 2
    }

    /// Number of phase collocation samples.
    #[must_use]
    pub const fn samples(&self) -> usize {
        self.samples
    }

    /// Synthesis weights for each sample, in coordinate order.
    pub(crate) fn rows(&self) -> impl ExactSizeIterator<Item = &[Real]> {
        self.basis.chunks_exact(self.coordinates())
    }

    pub(crate) fn row(&self, phase: usize) -> Option<&[Real]> {
        let width = self.coordinates();
        let start = phase.checked_mul(width)?;
        self.basis.get(start..start.checked_add(width)?)
    }

    /// Find the positive harmonic matching a physical source frequency.
    /// Returns its index in the positive-frequency list, not a coordinate.
    #[must_use]
    pub fn locate(&self, frequency: Real) -> Option<usize> {
        self.harmonics.iter().position(|harmonic| {
            (harmonic.frequency - frequency).abs()
                <= 8. * Real::EPSILON * harmonic.frequency.max(frequency.abs())
        })
    }

    /// Convert real coordinates to positive-frequency complex amplitudes
    /// (DC first). The convention is peak phasors, not half amplitudes:
    /// `x(t) = dc + Re(sum(amplitude * exp(j*omega*t)))`.
    ///
    /// # Errors
    /// Wrong coordinate count or nonfinite input.
    pub fn phasors(&self, coordinates: &[Real]) -> SpiceResult<Vec<Complex>> {
        self.check_coordinates(coordinates)?;
        let mut result = vec![Complex::real(coordinates[0])];
        result.extend(
            coordinates[1..]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|pair| Complex::new(pair[0], -pair[1])),
        );
        Ok(result)
    }

    fn check_coordinates(&self, coordinates: &[Real]) -> SpiceResult<()> {
        if coordinates.len() != self.coordinates()
            || coordinates.iter().any(|value| !value.is_finite())
        {
            return Err(failure("invalid harmonic coordinates"));
        }
        Ok(())
    }

    /// Synthesize samples from one real Fourier coordinate vector.
    ///
    /// # Errors
    /// Invalid shape or nonfinite result.
    pub fn synthesize(&self, coordinates: &[Real]) -> SpiceResult<Vec<Real>> {
        self.check_coordinates(coordinates)?;
        self.rows()
            .map(|row| {
                let value = row
                    .iter()
                    .zip(coordinates)
                    .fold(0., |sum, (&basis, &coefficient)| {
                        basis.mul_add(coefficient, sum)
                    });
                if value.is_finite() {
                    Ok(value)
                } else {
                    Err(failure("nonfinite harmonic synthesis"))
                }
            })
            .collect()
    }

    /// Orthogonally project real torus samples onto the retained coordinates.
    ///
    /// # Errors
    /// Invalid shape, nonfinite sample or projection.
    pub fn project(&self, samples: &[Real]) -> SpiceResult<Vec<Real>> {
        if samples.len() != self.samples || samples.iter().any(|value| !value.is_finite()) {
            return Err(failure("invalid harmonic sample vector"));
        }
        let count =
            Real::from(u32::try_from(self.samples).map_err(|_| failure("sample count overflow"))?);
        let mut result = vec![0.; self.coordinates()];
        for (row, &value) in self.rows().zip(samples) {
            let scaled = value / count;
            for (coefficient, &basis) in result.iter_mut().zip(row) {
                *coefficient = scaled.mul_add(basis, *coefficient);
            }
        }
        for value in &mut result[1..] {
            *value *= 2.;
        }
        self.check_coordinates(&result)?;
        Ok(result)
    }

    /// Apply the exact retained spectral time derivative. DC is zero;
    /// a cosine/sine pair `(a,b)` becomes `(omega*b,-omega*a)`.
    ///
    /// # Errors
    /// Invalid shape or nonfinite derivative.
    pub fn derivative(&self, coordinates: &[Real]) -> SpiceResult<Vec<Real>> {
        self.check_coordinates(coordinates)?;
        let mut result = vec![0.; self.coordinates()];
        for ((harmonic, pair), output) in self
            .harmonics
            .iter()
            .zip(coordinates[1..].as_chunks::<2>().0.iter())
            .zip(result[1..].as_chunks_mut::<2>().0.iter_mut())
        {
            let omega = TAU * harmonic.frequency;
            output[0] = omega * pair[1];
            output[1] = -omega * pair[0];
        }
        self.check_coordinates(&result)?;
        Ok(result)
    }
}

/// Typed, allocation-bounded analysis input. Unlike the executable lattice,
/// this retains only the harmonic indices and grid shape.
#[derive(Debug, Clone, PartialEq)]
pub struct HarmonicBalancePlan {
    shape: LatticeShape,
    limits: super::BalanceLimits,
}

impl HarmonicBalancePlan {
    pub(crate) fn resolve(
        arguments: &[String],
        oversampling: u16,
        iterations: usize,
    ) -> SpiceResult<Self> {
        if arguments.is_empty() || !arguments.as_chunks::<2>().1.is_empty() {
            return Err(failure(".hb requires frequency order pairs"));
        }
        let axes = arguments
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| {
                let frequency = crate::primitives::parse_spice_number(&pair[0])
                    .ok_or_else(|| failure(format!("invalid HB fundamental {}", pair[0])))?;
                let order = crate::analysis::plan::count(&pair[1], "harmonic order", 1, 512)?;
                let order = u16::try_from(order).map_err(|_| failure("harmonic order overflow"))?;
                Ok(ToneAxis { frequency, order })
            })
            .collect::<SpiceResult<Vec<_>>>()?;
        let shape = LatticeShape::new(&axes, oversampling, LatticeLimits::default())?;
        Ok(Self {
            shape,
            limits: super::BalanceLimits {
                iterations,
                ..super::BalanceLimits::default()
            },
        })
    }

    pub(crate) const fn grid_points(&self) -> usize {
        self.shape.samples
    }

    pub(crate) fn run(
        &self,
        circuit: &mut crate::devices::Circuit,
        context: &crate::analysis::AnalysisContext,
        newton: &crate::analysis::newton::NewtonOptions,
        sink: &mut dyn crate::analysis::PointSink,
    ) -> SpiceResult<()> {
        circuit.finalize()?;
        if !circuit.observations().is_empty() {
            return Err(failure(
                "HB device observations beyond MNA nodes and branch currents are not implemented",
            ));
        }
        let lattice = self.shape.materialize()?;
        // Residual equations have the opposite units to their corresponding
        // unknowns: KCL at a node (A), KVL/constitutive branch constraint (V).
        let mut tolerances = vec![newton.abstol; circuit.unknown_count()];
        for (index, _) in circuit.devices().iter().enumerate() {
            let range = circuit
                .branch_rows(index)
                .ok_or_else(|| failure("missing HB branch rows"))?;
            for tolerance in &mut tolerances[range] {
                *tolerance = newton.vntol;
            }
        }
        let result = super::solve_circuit(
            circuit,
            context.model_context(),
            &lattice,
            None,
            &tolerances,
            self.limits,
        )?;
        let header = crate::analysis::linear::plot(
            circuit,
            "hb",
            "Harmonic Balance",
            Some(("frequency", "frequency")),
            true,
        )?;
        sink.begin(&header)?;
        let m = lattice.coordinates();
        let mut row = Vec::with_capacity(circuit.unknown_count() + 1);
        for (index, frequency) in std::iter::once(0.)
            .chain(lattice.harmonics().iter().map(Harmonic::frequency))
            .enumerate()
        {
            row.clear();
            row.push(Complex::real(frequency));
            for coefficients in result.coefficients.as_slice().chunks_exact(m) {
                row.push(if index == 0 {
                    Complex::real(coefficients[0])
                } else {
                    Complex::new(coefficients[2 * index - 1], -coefficients[2 * index])
                });
            }
            sink.point(&row)?;
        }
        sink.end()
    }
}
