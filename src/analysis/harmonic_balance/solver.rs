//! Alternating-frequency/time Newton solve of F(x) + dQ(x)/dt = 0.
//! Each collocation point uses the same physical sparse MNA pattern. A
//! frequency-domain Jacobian expands its structural sites into harmonic
//! blocks, never into a dense all-unknowns matrix.

use std::sync::Arc;

use crate::maths::{Pattern, SparseLu, SparseMatrix, Vector};
use crate::primitives::{Complex, Real, SpiceError, SpiceResult, filled_vec};

use super::projection::Projection;

use super::{HarmonicLattice, failure};

/// Physical current/constraint and charge/flux equations at one phase point.
pub struct Sample {
    /// Static residual, including forcing.
    pub current: Vector,
    /// Dynamic quantity whose time derivative contributes to the residual.
    pub charge: Vector,
    /// Exact derivative of the static residual with respect to MNA unknowns.
    pub conductance: SparseMatrix,
    /// Exact derivative of the dynamic quantity with respect to MNA unknowns.
    pub capacitance: SparseMatrix,
}

/// An autonomous physical model evaluated under explicit periodic forcing.
/// The phase sample index is in the lattice's deterministic synthesis order.
pub trait HarmonicModel {
    /// Union of the static and dynamic Jacobian sites.
    fn pattern(&self) -> &Arc<Pattern>;

    /// Fill the zeroed sample buffers with physical values and derivatives.
    ///
    /// # Errors
    /// Invalid physical state, model or forcing.
    fn evaluate(&mut self, phase: usize, state: &Vector, sample: &mut Sample) -> SpiceResult<()>;
}

impl Sample {
    fn new(pattern: &Arc<Pattern>) -> SpiceResult<Self> {
        let n = pattern.order();
        Ok(Self {
            current: Vector::zeros(n),
            charge: Vector::zeros(n),
            conductance: SparseMatrix::bound(pattern.clone())?,
            capacitance: SparseMatrix::bound(pattern.clone())?,
        })
    }

    fn clear(&mut self) {
        self.current.clear();
        self.charge.clear();
        self.conductance.clear();
        self.capacitance.clear();
    }
}

/// Resource admission for one frequency-domain solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BalanceLimits {
    /// Maximum expanded real unknowns.
    pub unknowns: usize,
    /// Maximum structural Jacobian entries, before sparse LU fill.
    pub jacobian_entries: usize,
    /// Maximum worst-case dense factor entries, bounding possible fill.
    pub factor_entries: usize,
    /// Maximum retained phase samples per physical derivative field.
    pub sample_entries: usize,
    /// Maximum cumulative scalar assembly work.
    pub assembly_work: usize,
    /// Maximum full nonlinear model evaluations (one per phase sample).
    pub model_evaluations: usize,
    /// Maximum Newton corrections.
    pub iterations: usize,
    /// Maximum line-search trials per correction.
    pub line_search: usize,
}

impl Default for BalanceLimits {
    fn default() -> Self {
        Self {
            unknowns: 4096,
            jacobian_entries: 2_000_000,
            factor_entries: 16_000_000,
            sample_entries: 8_000_000,
            assembly_work: 250_000_000,
            model_evaluations: 100_000,
            iterations: 40,
            line_search: 16,
        }
    }
}

/// Solver work accounting and final physical residual.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BalanceStats {
    /// Full physical-model evaluations.
    pub model_evaluations: usize,
    /// Charged scalar assembly work.
    pub assembly_work: usize,
    /// Peak allocated sparse LU element slots.
    pub peak_factor_entries: usize,
    /// Newton linear solves.
    pub corrections: usize,
    /// Rejected line-search candidates.
    pub rejected: usize,
    /// Maximum residual normalized by the supplied equation tolerances.
    pub scaled_residual: Real,
}

/// Converged real Fourier coordinates, stored by physical MNA row.
pub struct BalanceSolution {
    /// Row-major real Fourier state.
    pub coefficients: Vector,
    /// Full spectral residual in the same order.
    pub residual: Vector,
    /// Solver work and convergence diagnostics.
    pub stats: BalanceStats,
}

struct Workspace<'a> {
    lattice: &'a HarmonicLattice,
    local: Arc<Pattern>,
    sample: Sample,
    state: Vector,
    matrix: Option<SparseMatrix>,
    blocks: Vec<(usize, usize)>,
    projection: Projection,
    conductance: Vec<Real>,
    capacitance: Vec<Real>,
    g_spectrum: Vec<Complex>,
    c_spectrum: Vec<Complex>,
    block: Vec<Real>,
    weights: Vec<Real>,
    charge_offset: Vec<Real>,
    derivative_weights: Vec<Real>,
    tolerance: &'a [Real],
    remaining_work: usize,
    remaining_evaluations: usize,
    assembly_cost: usize,
    residual_cost: usize,
}

fn product(a: usize, b: usize, limit: usize, message: &str) -> SpiceResult<usize> {
    a.checked_mul(b)
        .filter(|count| *count <= limit)
        .ok_or_else(|| failure(message))
}

impl<'a> Workspace<'a> {
    fn new(
        lattice: &'a HarmonicLattice,
        local: Arc<Pattern>,
        tolerance: &'a [Real],
        limits: BalanceLimits,
        expand_newton: bool,
    ) -> SpiceResult<Self> {
        let n = local.order();
        let m = lattice.coordinates();
        if n == 0 || tolerance.len() != n || tolerance.iter().any(|x| !x.is_finite() || *x <= 0.) {
            return Err(failure("invalid harmonic balance equation tolerances"));
        }
        let dimension = product(n, m, limits.unknowns, "harmonic unknown limit exceeded")?;
        product(
            dimension,
            dimension,
            limits.factor_entries,
            "harmonic factor fill limit exceeded",
        )?;
        let block_size = product(
            m,
            m,
            limits.jacobian_entries,
            "harmonic block limit exceeded",
        )?;
        let entries = product(
            local.len(),
            block_size,
            limits.jacobian_entries,
            "harmonic Jacobian limit exceeded",
        )?;
        let samples = lattice.samples();
        let sample_entries = product(
            local.len(),
            samples,
            limits.sample_entries,
            "physical derivative sample limit exceeded",
        )?;
        let projection = Projection::new(lattice, limits.assembly_work, limits.sample_entries)?;
        let residual_cost = product(
            dimension,
            3,
            limits.assembly_work,
            "harmonic residual work limit exceeded",
        )?;
        let residual_cost = product(
            residual_cost,
            samples,
            limits.assembly_work,
            "harmonic residual work limit exceeded",
        )?;
        let transform_cost = projection
            .cost()
            .checked_mul(2)
            .and_then(|cost| cost.checked_mul(local.len()))
            .ok_or_else(|| failure("harmonic projection work overflow"))?;
        let assembly_cost = residual_cost
            .checked_add(transform_cost)
            .and_then(|cost| cost.checked_add(entries))
            .filter(|&cost| cost <= limits.assembly_work)
            .ok_or_else(|| failure("harmonic assembly work limit exceeded"))?;
        let matrix = if expand_newton {
            let coordinates = local.positions().flat_map(|(r, c)| {
                (0..m).flat_map(move |i| (0..m).map(move |j| (r * m + i, c * m + j)))
            });
            Some(SparseMatrix::bound(Arc::new(Pattern::new(
                dimension,
                coordinates,
            )?))?)
        } else {
            None
        };
        let mut blocks = Vec::new();
        blocks
            .try_reserve_exact(local.len())
            .map_err(|e| failure(e.to_string()))?;
        for (r, c) in local.positions() {
            blocks.push((r, c));
        }
        let size = projection.size();
        Ok(Self {
            lattice,
            sample: Sample::new(&local)?,
            state: Vector::zeros(n),
            matrix,
            local,
            blocks,
            projection,
            conductance: filled_vec(sample_entries, 0., "physical conductance samples")?,
            capacitance: filled_vec(sample_entries, 0., "physical capacitance samples")?,
            g_spectrum: filled_vec(size, Complex::ZERO, "conductance spectrum")?,
            c_spectrum: filled_vec(size, Complex::ZERO, "capacitance spectrum")?,
            block: filled_vec(
                if expand_newton { block_size } else { 0 },
                0.,
                "harmonic Jacobian block",
            )?,
            weights: vec![0.; m],
            charge_offset: vec![0.; n],
            derivative_weights: vec![0.; m],
            tolerance,
            remaining_work: limits.assembly_work,
            remaining_evaluations: limits.model_evaluations,
            assembly_cost,
            residual_cost,
        })
    }

    fn evaluate(
        &mut self,
        model: &mut impl HarmonicModel,
        coefficients: &Vector,
        jacobian: bool,
    ) -> SpiceResult<Vector> {
        let m = self.lattice.coordinates();
        let count = self.lattice.samples();
        let cost = if jacobian {
            self.assembly_cost
        } else {
            self.residual_cost
        };
        self.remaining_work = self
            .remaining_work
            .checked_sub(cost)
            .ok_or_else(|| SpiceError::circuit("cumulative harmonic assembly work exhausted"))?;
        self.remaining_evaluations = self
            .remaining_evaluations
            .checked_sub(count)
            .ok_or_else(|| SpiceError::circuit("harmonic model evaluation budget exhausted"))?;
        let normalization =
            1. / Real::from(u32::try_from(count).map_err(|_| failure("phase count overflow"))?);
        let mut residual = Vector::zeros(coefficients.len());
        for (phase, basis) in self.lattice.rows().enumerate() {
            for (value, row) in self
                .state
                .as_mut_slice()
                .iter_mut()
                .zip(coefficients.as_slice().chunks_exact(m))
            {
                *value = row
                    .iter()
                    .zip(basis)
                    .fold(0., |sum, (&a, &b)| a.mul_add(b, sum));
            }
            if !self.state.is_finite() {
                return Err(failure("nonfinite synthesized state"));
            }
            self.sample.clear();
            model.evaluate(phase, &self.state, &mut self.sample)?;
            if !self.sample.current.is_finite()
                || !self.sample.charge.is_finite()
                || self.sample.current.len() != self.local.order()
                || self.sample.charge.len() != self.local.order()
            {
                return Err(failure("invalid physical harmonic sample"));
            }
            if phase == 0 {
                self.charge_offset
                    .copy_from_slice(self.sample.charge.as_slice());
            }
            self.weights[0] = normalization;
            self.derivative_weights[0] = 0.;
            for (index, harmonic) in self.lattice.harmonics().iter().enumerate() {
                let i = 1 + 2 * index;
                let omega = std::f64::consts::TAU * harmonic.frequency();
                self.weights[i] = 2. * normalization * basis[i];
                self.weights[i + 1] = 2. * normalization * basis[i + 1];
                self.derivative_weights[i] = omega * self.weights[i + 1];
                self.derivative_weights[i + 1] = -omega * self.weights[i];
            }
            for ((row, (&current, &charge)), &offset) in residual
                .as_mut_slice()
                .chunks_exact_mut(m)
                .zip(
                    self.sample
                        .current
                        .as_slice()
                        .iter()
                        .zip(self.sample.charge.as_slice()),
                )
                .zip(&self.charge_offset)
            {
                for ((value, &w), &dw) in row
                    .iter_mut()
                    .zip(&self.weights)
                    .zip(&self.derivative_weights)
                {
                    *value = (charge - offset).mul_add(dw, current.mul_add(w, *value));
                }
            }
            if jacobian {
                self.record_derivatives(phase)?;
            }
        }
        if !residual.is_finite() {
            return Err(failure("nonfinite spectral residual"));
        }
        if jacobian && self.matrix.is_some() {
            self.publish_matrix()?;
        }
        Ok(residual)
    }

    fn record_derivatives(&mut self, phase: usize) -> SpiceResult<()> {
        let (g_pattern, g) = self
            .sample
            .conductance
            .bound_values()
            .ok_or_else(|| failure("static model Jacobian exceeded declared pattern"))?;
        let (c_pattern, c) = self
            .sample
            .capacitance
            .bound_values()
            .ok_or_else(|| failure("dynamic model Jacobian exceeded declared pattern"))?;
        if !Arc::ptr_eq(g_pattern, &self.local) || !Arc::ptr_eq(c_pattern, &self.local) {
            return Err(failure("model replaced its declared Jacobian pattern"));
        }
        let count = self.lattice.samples();
        for (site, (&g, &c)) in g.iter().zip(c).enumerate() {
            if !g.is_finite() || !c.is_finite() {
                return Err(failure("nonfinite model Jacobian"));
            }
            self.conductance[site * count + phase] = g;
            self.capacitance[site * count + phase] = c;
        }
        Ok(())
    }

    fn publish_matrix(&mut self) -> SpiceResult<()> {
        let matrix = self
            .matrix
            .as_mut()
            .ok_or_else(|| failure("missing Newton Jacobian workspace"))?;
        matrix.clear();
        let m = self.lattice.coordinates();
        let count = self.lattice.samples();
        for (site, &(row, col)) in self.blocks.iter().enumerate() {
            let range = site * count..(site + 1) * count;
            self.projection
                .transform(&self.conductance[range.clone()], &mut self.g_spectrum);
            self.projection
                .transform(&self.capacitance[range], &mut self.c_spectrum);
            self.projection.block(
                self.lattice,
                &self.g_spectrum,
                &self.c_spectrum,
                &mut self.block,
            );
            for (i, values) in self.block.chunks_exact(m).enumerate() {
                for (j, &value) in values.iter().enumerate() {
                    let scaled = value / self.tolerance[row];
                    if !scaled.is_finite() {
                        return Err(failure("nonfinite scaled harmonic Jacobian"));
                    }
                    matrix.add(row * m + i, col * m + j, scaled)?;
                }
            }
        }
        Ok(())
    }

    fn scaled(&self, residual: &Vector) -> SpiceResult<Vector> {
        let m = self.lattice.coordinates();
        let scaled = Vector::from(
            residual
                .as_slice()
                .iter()
                .enumerate()
                .map(|(i, x)| x / self.tolerance[i / m])
                .collect::<Vec<_>>(),
        );
        if !scaled.is_finite() {
            return Err(failure("nonfinite scaled harmonic residual"));
        }
        Ok(scaled)
    }
}

/// Physical derivative spectra on the complete difference lattice. The
/// expanded pattern is shared by all shifted-frequency operators.
pub(in crate::analysis) struct OrbitSpectrum {
    pub(in crate::analysis) pattern: Arc<Pattern>,
    pub(in crate::analysis) unknowns: usize,
    pub(in crate::analysis) width: usize,
    pub(in crate::analysis) spectral_indices: Vec<usize>,
    pub(in crate::analysis) conductance: Vec<Complex>,
    pub(in crate::analysis) capacitance: Vec<Complex>,
    pub(in crate::analysis) scaled_residual: Real,
}

/// Recheck the actual supplied orbit under the current model, and project
/// its exact derivatives once. A caller-provided convergence flag alone
/// cannot authorize a linearization.
pub(in crate::analysis) fn linearize_orbit(
    lattice: &HarmonicLattice,
    model: &mut impl HarmonicModel,
    solution: &BalanceSolution,
    tolerance: &[Real],
    limits: BalanceLimits,
) -> SpiceResult<OrbitSpectrum> {
    let mut work = Workspace::new(lattice, model.pattern().clone(), tolerance, limits, false)?;
    if solution.coefficients.len() != work.local.order() * lattice.coordinates()
        || !solution.coefficients.is_finite()
    {
        return Err(failure("invalid periodic reference shape or values"));
    }
    let residual = work.evaluate(model, &solution.coefficients, true)?;
    let scaled_residual = work.scaled(&residual)?.max_abs();
    if scaled_residual > 1. {
        return Err(failure(format!(
            "periodic reference does not satisfy the physical equations: {scaled_residual:.6e}"
        )));
    }
    let size = work.projection.size();
    let count = product(
        work.local.len(),
        size,
        limits.sample_entries,
        "periodic derivative spectrum budget exceeded",
    )?;
    let mut conductance = filled_vec(count, Complex::ZERO, "periodic conductance spectra")?;
    let mut capacitance = filled_vec(count, Complex::ZERO, "periodic capacitance spectra")?;
    let phases = lattice.samples();
    for (site, (g, c)) in conductance
        .chunks_exact_mut(size)
        .zip(capacitance.chunks_exact_mut(size))
        .enumerate()
    {
        let range = site * phases..(site + 1) * phases;
        work.projection
            .transform(&work.conductance[range.clone()], g);
        work.projection.transform(&work.capacitance[range], c);
        if g.iter().chain(c.iter()).any(|value| !value.is_finite()) {
            return Err(failure("nonfinite periodic derivative spectrum"));
        }
    }
    let differences = work.projection.signed_differences()?;
    let width = lattice.coordinates();
    let mut positions = Vec::new();
    let mut spectral_indices = Vec::new();
    for (site, (row, col)) in work.local.positions().enumerate() {
        let offset = site * size;
        let supports = |index: usize| {
            conductance[offset + index] != Complex::ZERO
                || capacitance[offset + index] != Complex::ZERO
        };
        let support_count = differences.iter().filter(|&&index| supports(index)).count();
        positions
            .try_reserve(support_count)
            .map_err(|e| failure(e.to_string()))?;
        spectral_indices
            .try_reserve(support_count)
            .map_err(|e| failure(e.to_string()))?;
        for (ordinal, &index) in differences.iter().enumerate() {
            // Exact structural zero only: no accuracy threshold on small
            // mixing coefficients. Constant derivatives retain just the
            // diagonal sideband blocks.
            if supports(index) {
                positions.push((row * width + ordinal / width, col * width + ordinal % width));
                spectral_indices.push(offset + index);
            }
        }
    }
    let pattern = Arc::new(Pattern::new(work.local.order() * width, positions)?);
    Ok(OrbitSpectrum {
        pattern,
        unknowns: work.local.order(),
        width,
        spectral_indices,
        conductance,
        capacitance,
        scaled_residual,
    })
}

/// Solve the coupled forced frequency-domain equations with analytic sparse
/// Newton derivatives and a residual-decreasing line search.
///
/// The caller supplies physical equation tolerances, one per MNA row, and
/// a real Fourier seed with one contiguous coordinate vector per row.
/// Success requires every retained spectral equation to meet its tolerance.
///
/// # Errors
/// Invalid model, shapes, resource limits, failed sparse solve or nonlinear
/// nonconvergence. There is no transient fallback.
pub fn solve(
    lattice: &HarmonicLattice,
    model: &mut impl HarmonicModel,
    mut coefficients: Vector,
    tolerance: &[Real],
    limits: BalanceLimits,
) -> SpiceResult<BalanceSolution> {
    let mut work = Workspace::new(lattice, model.pattern().clone(), tolerance, limits, true)?;
    if coefficients.len() != work.local.order() * lattice.coordinates()
        || !coefficients.is_finite()
        || limits.iterations == 0
        || limits.line_search == 0
    {
        return Err(failure("invalid harmonic Newton seed or iteration limits"));
    }
    let mut factor: Option<SparseLu> = None;
    let mut rejected = 0;
    let mut factor_workspace = Vec::new();
    let mut delta = Vector::default();
    let mut rhs = Vector::default();
    let mut peak_factor_entries = 0;
    for corrections in 0..=limits.iterations {
        let residual = work.evaluate(model, &coefficients, true)?;
        let scaled = work.scaled(&residual)?;
        let norm = scaled.max_abs();
        if norm <= 1. {
            return Ok(BalanceSolution {
                coefficients,
                residual,
                stats: BalanceStats {
                    model_evaluations: limits.model_evaluations - work.remaining_evaluations,
                    assembly_work: limits.assembly_work - work.remaining_work,
                    peak_factor_entries,
                    corrections,
                    rejected,
                    scaled_residual: norm,
                },
            });
        }
        if corrections == limits.iterations {
            break;
        }
        let matrix = work
            .matrix
            .as_ref()
            .ok_or_else(|| failure("missing Newton Jacobian workspace"))?;
        let next = SparseLu::new(matrix, factor.as_ref().map(SparseLu::symbolic))?;
        rhs.copy_values_from(&scaled);
        for value in rhs.as_mut_slice() {
            *value = -*value;
        }
        next.solve_into(&rhs, &mut factor_workspace, &mut delta, false)?;
        peak_factor_entries = peak_factor_entries.max(next.factor_storage().factor_entries);
        factor = Some(next);
        let mut accepted = false;
        let mut step: Real = 1.;
        for _ in 0..limits.line_search {
            let candidate = Vector::from(
                coefficients
                    .as_slice()
                    .iter()
                    .zip(delta.as_slice())
                    .map(|(&x, &dx)| step.mul_add(dx, x))
                    .collect::<Vec<_>>(),
            );
            if candidate.is_finite() {
                match work.evaluate(model, &candidate, false) {
                    Ok(trial) if work.scaled(&trial)?.max_abs() < norm => {
                        coefficients = candidate;
                        accepted = true;
                        break;
                    }
                    Ok(_) | Err(crate::primitives::SpiceError::Numerical { .. }) => {}
                    Err(error) => return Err(error),
                }
            }
            rejected += 1;
            step *= 0.5;
        }
        if !accepted {
            return Err(failure(format!(
                "harmonic Newton line search failed at scaled residual {norm:.6e}"
            )));
        }
    }
    Err(failure("harmonic Newton iteration limit exhausted"))
}
