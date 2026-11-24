//! Alternating-frequency/time Newton solve of F(x) + dQ(x)/dt = 0.
//! Each collocation point uses the same physical sparse MNA pattern. A
//! frequency-domain Jacobian expands its structural sites into harmonic
//! blocks, never into a dense all-unknowns matrix.

use std::sync::Arc;

use crate::maths::{Pattern, SparseLu, SparseMatrix, Vector};
use crate::primitives::{Real, SpiceResult, filled_vec};

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
    matrix: SparseMatrix,
    blocks: Vec<(usize, usize)>,
    weights: Vec<Real>,
    derivative_weights: Vec<Real>,
    tolerance: &'a [Real],
    remaining_work: usize,
    remaining_evaluations: usize,
    assembly_cost: usize,
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
        let assembly_cost = entries
            .checked_add(
                dimension
                    .checked_mul(3)
                    .ok_or_else(|| failure("harmonic work overflow"))?,
            )
            .ok_or_else(|| failure("harmonic work overflow"))?;
        product(
            assembly_cost,
            lattice.samples(),
            limits.assembly_work,
            "harmonic assembly work limit exceeded",
        )?;
        let coordinates = local.positions().flat_map(|(r, c)| {
            (0..m).flat_map(move |i| (0..m).map(move |j| (r * m + i, c * m + j)))
        });
        let expanded = Arc::new(Pattern::new(dimension, coordinates)?);
        let mut blocks = Vec::new();
        blocks
            .try_reserve_exact(local.len())
            .map_err(|e| failure(e.to_string()))?;
        for (r, c) in local.positions() {
            blocks.push((r, c));
        }
        Ok(Self {
            lattice,
            sample: Sample::new(&local)?,
            state: Vector::zeros(n),
            matrix: SparseMatrix::bound(expanded)?,
            local,
            blocks,
            weights: vec![0.; m],
            derivative_weights: vec![0.; m],
            tolerance,
            remaining_work: limits.assembly_work,
            remaining_evaluations: limits.model_evaluations,
            assembly_cost,
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
            coefficients.len().saturating_mul(3)
        };
        self.remaining_work = self
            .remaining_work
            .checked_sub(
                cost.checked_mul(count)
                    .ok_or_else(|| failure("harmonic work overflow"))?,
            )
            .ok_or_else(|| failure("cumulative harmonic assembly work exhausted"))?;
        self.remaining_evaluations = self
            .remaining_evaluations
            .checked_sub(count)
            .ok_or_else(|| failure("harmonic model evaluation budget exhausted"))?;
        let normalization =
            1. / Real::from(u32::try_from(count).map_err(|_| failure("phase count overflow"))?);
        let mut residual = Vector::zeros(coefficients.len());
        let mut entries = if jacobian {
            filled_vec(self.blocks.len() * m * m, 0., "harmonic Jacobian blocks")?
        } else {
            Vec::new()
        };
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
            for (row, (&current, &charge)) in residual.as_mut_slice().chunks_exact_mut(m).zip(
                self.sample
                    .current
                    .as_slice()
                    .iter()
                    .zip(self.sample.charge.as_slice()),
            ) {
                for ((value, &w), &dw) in row
                    .iter_mut()
                    .zip(&self.weights)
                    .zip(&self.derivative_weights)
                {
                    *value = charge.mul_add(dw, current.mul_add(w, *value));
                }
            }
            if jacobian {
                self.accumulate_blocks(basis, &mut entries)?;
            }
        }
        if !residual.is_finite() {
            return Err(failure("nonfinite spectral residual"));
        }
        if jacobian {
            self.publish_matrix(&entries)?;
        }
        Ok(residual)
    }

    fn accumulate_blocks(&self, basis: &[Real], entries: &mut [Real]) -> SpiceResult<()> {
        let m = basis.len();
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
        for ((block, &g), &c) in entries.chunks_exact_mut(m * m).zip(g).zip(c) {
            if !g.is_finite() || !c.is_finite() {
                return Err(failure("nonfinite model Jacobian"));
            }
            if g == 0. && c == 0. {
                continue;
            }
            for ((row, &w), &dw) in block
                .chunks_exact_mut(m)
                .zip(&self.weights)
                .zip(&self.derivative_weights)
            {
                let value = c.mul_add(dw, g * w);
                for (entry, &b) in row.iter_mut().zip(basis) {
                    *entry = value.mul_add(b, *entry);
                }
            }
        }
        Ok(())
    }

    fn publish_matrix(&mut self, entries: &[Real]) -> SpiceResult<()> {
        self.matrix.clear();
        let m = self.lattice.coordinates();
        for (&(row, col), block) in self.blocks.iter().zip(entries.chunks_exact(m * m)) {
            for (i, values) in block.chunks_exact(m).enumerate() {
                for (j, &value) in values.iter().enumerate() {
                    let scaled = value / self.tolerance[row];
                    if !scaled.is_finite() {
                        return Err(failure("nonfinite scaled harmonic Jacobian"));
                    }
                    self.matrix.add(row * m + i, col * m + j, scaled)?;
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
    let mut work = Workspace::new(lattice, model.pattern().clone(), tolerance, limits)?;
    if coefficients.len() != work.matrix.rows()
        || !coefficients.is_finite()
        || limits.iterations == 0
        || limits.line_search == 0
    {
        return Err(failure("invalid harmonic Newton seed or iteration limits"));
    }
    let mut factor: Option<SparseLu> = None;
    let mut rejected = 0;
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
                    corrections,
                    rejected,
                    scaled_residual: norm,
                },
            });
        }
        if corrections == limits.iterations {
            break;
        }
        let next = SparseLu::new(&work.matrix, factor.as_ref().map(SparseLu::symbolic))?;
        let rhs = Vector::from(
            scaled
                .as_slice()
                .iter()
                .map(|value| -value)
                .collect::<Vec<_>>(),
        );
        let delta = next.solve(&rhs)?;
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
