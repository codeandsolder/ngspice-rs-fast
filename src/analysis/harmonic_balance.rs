//! Native multi-tone harmonic-balance residual and Newton solver.
//!
//! Algebraic MNA rows, linear charge storage, nonlinear conduction and
//! nonlinear charge use the same operator for DC and every Fourier mode:
//!     G*x + i(x) + d/dt (C*x + q(x)) - independent_forcing = 0.
//! No transient startup, Volterra approximation or fixed-frequency oscillator
//! phase assumption is substituted for this coupled frequency-domain solve.
//!
//! The nonlinear callback is a contribution collector, not a replacement
//! source of model physics. Circuit-device adapters must supply both its
//! currents/charges and exact local Jacobian contributions.
pub mod plan;

use crate::maths::dense::{Matrix, Vector};
use crate::primitives::{Complex, Real, SpiceError, SpiceResult};
use plan::HarmonicPlan;
use std::f64::consts::TAU;

fn numerical(message: impl Into<String>) -> SpiceError {
    SpiceError::Numerical {
        context: "harmonic balance".to_owned(),
        message: message.into(),
    }
}

/// Nonlinear conductive and stored-charge MNA contributions at one real
/// torus sample, including their exact partial derivatives.
///
/// Each call receives zeroed output slices. The caller must return all
/// applicable nonlinear device contributions in the usual signed MNA rows;
/// no simplified model or finite-difference derivative is implied.
pub trait NonlinearConstitutive {
    /// Evaluate currents, stored charges and their Jacobians at a real state.
    ///
    /// Jacobians are n×n row-major matrices. Pure capacitors and conductors
    /// may instead use the fixed matrices supplied to the problem.
    ///
    /// # Errors
    /// Invalid physical state or non-finite model coefficients.
    fn evaluate(
        &mut self,
        state: &[Real],
        currents: &mut [Real],
        charges: &mut [Real],
        conductance: &mut [Real],
        capacitance: &mut [Real],
    ) -> SpiceResult<()>;
}

/// Numerical resource and convergence policy for one nonlinear spectrum.
#[derive(Debug, Clone, Copy)]
pub struct SolveLimits {
    /// Maximum real MNA unknowns (including voltage-source branches).
    pub unknowns: usize,
    /// Maximum retained real degrees of freedom for dense exact Newton.
    pub dense_dofs: usize,
    /// Conservative upper bound on analytical Jv and DFT products.
    pub work_units: usize,
    /// Number of Newton updates before a structured convergence failure.
    pub iterations: usize,
    /// Maximum halving attempts for each Newton step.
    pub line_search_steps: usize,
    /// Relative residual tolerance applied per physical equation.
    pub relative: Real,
    /// Absolute tolerance in physical row units.
    pub absolute: Real,
}

impl Default for SolveLimits {
    fn default() -> Self {
        Self {
            unknowns: 256,
            dense_dofs: 256,
            work_units: 250_000_000,
            iterations: 30,
            line_search_steps: 16,
            relative: 1e-10,
            absolute: 1e-13,
        }
    }
}

/// Retained converged full spectrum, including negative-frequency conjugates.
#[derive(Debug, Clone)]
pub struct HarmonicSolution {
    /// Mode-major [harmonic][MNA unknown], complex Fourier convention.
    pub spectrum: Vec<Complex>,
    /// Per-real-row maximum normalized nonlinear KCL/charge residual.
    pub residual_ratio: Real,
    pub iterations: usize,
    pub sampled_model_calls: usize,
}

/// A frequency-domain circuit with linear MNA operators and nonlinear laws.
///
/// Linear G and C are both row-major n×n, and forcing is stored mode-major.
/// The nonlinear local Jacobians are evaluated once per torus point per
/// Newton attempt, then reused across all matrix-free directional actions.
pub struct HarmonicProblem {
    plan: HarmonicPlan,
    unknowns: usize,
    g: Vec<Real>,
    c: Vec<Real>,
    forcing: Vec<Complex>,
    limits: SolveLimits,
}

#[derive(Clone, Copy)]
enum Part {
    Real,
    Imaginary,
}

#[derive(Clone, Copy)]
struct Coordinate {
    mode: usize,
    row: usize,
    part: Part,
}

/// Real degrees of freedom for a conjugate-symmetric complex spectrum.
/// A self-conjugate DC mode contributes one real coordinate; every other
/// mode pair contributes real and imaginary components for each MNA row.
struct Coordinates {
    independent: Vec<Coordinate>,
    unknowns: usize,
    mode_count: usize,
}

impl Coordinates {
    fn new(plan: &HarmonicPlan, unknowns: usize) -> Self {
        let mut independent = Vec::with_capacity(unknowns * plan.modes().len());
        for (mode, harmonic) in plan.modes().iter().enumerate() {
            if mode > harmonic.conjugate {
                continue;
            }
            for row in 0..unknowns {
                independent.push(Coordinate {
                    mode,
                    row,
                    part: Part::Real,
                });
                if mode != harmonic.conjugate {
                    independent.push(Coordinate {
                        mode,
                        row,
                        part: Part::Imaginary,
                    });
                }
            }
        }
        Self {
            independent,
            unknowns,
            mode_count: plan.modes().len(),
        }
    }

    fn decode(&self, plan: &HarmonicPlan, real: &[Real]) -> Vec<Complex> {
        let mut full = vec![Complex::ZERO; self.mode_count * self.unknowns];
        for (&value, coordinate) in real.iter().zip(&self.independent) {
            let partner = plan.modes()[coordinate.mode].conjugate;
            let cell = coordinate.mode * self.unknowns + coordinate.row;
            let opposite = partner * self.unknowns + coordinate.row;
            match coordinate.part {
                Part::Real => {
                    full[cell].re = value;
                    full[opposite].re = value;
                }
                Part::Imaginary => {
                    full[cell].im = value;
                    full[opposite].im = -value;
                }
            }
        }
        full
    }

    fn encode(&self, full: &[Complex]) -> Vec<Real> {
        self.independent
            .iter()
            .map(|coord| {
                let cell = full[coord.mode * self.unknowns + coord.row];
                match coord.part {
                    Part::Real => cell.re,
                    Part::Imaginary => cell.im,
                }
            })
            .collect()
    }
}

struct Sampled {
    di: Vec<Real>,
    dq: Vec<Real>,
    ispectrum: Vec<Complex>,
    qspectrum: Vec<Complex>,
    calls: usize,
}

impl HarmonicProblem {
    /// Construct a constrained nonlinear frequency-domain problem.
    ///
    /// # Errors
    /// Invalid dimensions, samples/unknown work, nonconjugate forcing,
    /// nonfinite linear stamps, or invalid convergence settings.
    pub fn new(
        plan: HarmonicPlan,
        unknowns: usize,
        g: Vec<Real>,
        c: Vec<Real>,
        forcing: Vec<Complex>,
        limits: SolveLimits,
    ) -> SpiceResult<Self> {
        if unknowns == 0 || unknowns > limits.unknowns {
            return Err(numerical("invalid or over-budget MNA unknown count"));
        }
        let size = unknowns
            .checked_mul(unknowns)
            .ok_or_else(|| numerical("MNA matrix dimension overflow"))?;
        let variables = unknowns
            .checked_mul(plan.modes().len())
            .ok_or_else(|| numerical("spectral MNA dimension overflow"))?;
        if g.len() != size
            || c.len() != size
            || forcing.len() != variables
            || g.iter().chain(&c).any(|v| !v.is_finite())
            || forcing.iter().any(|v| !v.is_finite())
        {
            return Err(numerical("nonfinite or incorrectly shaped MNA input"));
        }
        if variables > limits.dense_dofs {
            return Err(numerical(
                "dense Newton dimensions exceed the configured budget",
            ));
        }
        if !limits.relative.is_finite()
            || !limits.absolute.is_finite()
            || limits.relative <= 0.
            || limits.absolute <= 0.
            || limits.iterations == 0
            || limits.line_search_steps == 0
        {
            return Err(numerical("invalid Newton tolerance or iteration budget"));
        }
        // Upper bound for one full analytic Newton cycle, with the
        // multidimensional direct DFT used for every Jv basis action.
        let work = variables
            .checked_mul(variables)
            .and_then(|value| value.checked_mul(plan.sample_count()))
            .and_then(|value| value.checked_mul(unknowns))
            .ok_or_else(|| numerical("harmonic Newton work product overflow"))?;
        if work > limits.work_units {
            return Err(numerical(
                "harmonic Newton work exceeds the configured budget",
            ));
        }
        for row in 0..unknowns {
            let spectral: Vec<_> = forcing
                .chunks_exact(unknowns)
                .map(|point| point[row])
                .collect();
            plan.synthesize(&spectral)?;
        }
        Ok(Self {
            plan,
            unknowns,
            g,
            c,
            forcing,
            limits,
        })
    }

    #[must_use]
    pub const fn plan(&self) -> &HarmonicPlan {
        &self.plan
    }

    /// Evaluate a complete multi-tone, fully nonlinear forced steady state.
    ///
    /// A first guess can be supplied as a full conjugate-symmetric spectrum;
    /// None starts with zero state. Explicit autonomous frequency and phase
    /// unknowns are not assumed by this forced-frequency solver.
    ///
    /// # Errors
    /// Singular Jacobian, invalid device evaluation or failure to reach
    /// the specified residual tolerance within the bounded Newton steps.
    pub fn solve(
        &self,
        nonlinear: &mut impl NonlinearConstitutive,
        initial: Option<&[Complex]>,
    ) -> SpiceResult<HarmonicSolution> {
        let coords = Coordinates::new(&self.plan, self.unknowns);
        let dofs = coords.independent.len();
        let mut x = if let Some(initial) = initial {
            if initial.len() != self.forcing.len() {
                return Err(numerical("initial harmonic spectrum has wrong dimensions"));
            }
            for row in 0..self.unknowns {
                let values: Vec<_> = initial
                    .chunks_exact(self.unknowns)
                    .map(|mode| mode[row])
                    .collect();
                self.plan.synthesize(&values)?;
            }
            coords.encode(initial)
        } else {
            vec![0.; dofs]
        };
        let mut model_calls = 0;
        let mut work = self.evaluate(nonlinear, &coords, &x)?;
        model_calls += work.1.calls;
        for iteration in 0..=self.limits.iterations {
            let ratio = self.residual_ratio(&work.0, &work.1);
            if ratio <= 1. {
                return Ok(HarmonicSolution {
                    spectrum: coords.decode(&self.plan, &x),
                    residual_ratio: ratio,
                    iterations: iteration,
                    sampled_model_calls: model_calls,
                });
            }
            if iteration == self.limits.iterations {
                return Err(numerical(format!(
                    "nonconvergent forced harmonic balance: normalized residual {ratio:.6e} after {iteration} Newton steps"
                )));
            }
            let mut jac = Matrix::zeros(dofs, dofs)?;
            let mut direction = vec![0.; dofs];
            for col in 0..dofs {
                direction[col] = 1.;
                let result = self.jacobian_action(&coords, &work.1, &direction)?;
                for (row, value) in result.iter().enumerate() {
                    jac.set(row, col, *value)?;
                }
                direction[col] = 0.;
            }
            let dx = jac.solve(&Vector::from_slice(
                &work.0.iter().map(|value| -value).collect::<Vec<_>>(),
            ))?;
            let mut accepted = None;
            let mut alpha = 1.;
            for _ in 0..self.limits.line_search_steps {
                let trial: Vec<_> = x
                    .iter()
                    .zip(dx.as_slice())
                    .map(|(value, delta)| value + alpha * delta)
                    .collect();
                if trial.iter().all(|value| value.is_finite()) {
                    if let Ok(next) = self.evaluate(nonlinear, &coords, &trial) {
                        model_calls += next.1.calls;
                        let merit = self.residual_ratio(&next.0, &next.1);
                        if merit.is_finite() && merit < ratio * (1. - 1e-4 * alpha) {
                            accepted = Some((trial, next));
                            break;
                        }
                    }
                }
                alpha *= 0.5;
            }
            let (next_x, next_work) = accepted
                .ok_or_else(|| numerical("no admissible residual-decreasing Newton step"))?;
            x = next_x;
            work = next_work;
        }
        Err(numerical("unreachable harmonic balance iteration state"))
    }

    fn evaluate(
        &self,
        nonlinear: &mut impl NonlinearConstitutive,
        coords: &Coordinates,
        x: &[Real],
    ) -> SpiceResult<(Vec<Real>, Sampled)> {
        let spectrum = coords.decode(&self.plan, x);
        let m = self.plan.modes().len();
        let s = self.plan.sample_count();
        let n = self.unknowns;
        let mut state = vec![0.; n * s];
        for row in 0..n {
            let coefficients: Vec<_> = spectrum.chunks_exact(n).map(|point| point[row]).collect();
            let values = self.plan.synthesize(&coefficients)?;
            for (sample, value) in values.into_iter().enumerate() {
                state[sample * n + row] = value;
            }
        }
        let mut currents = vec![0.; n * s];
        let mut charges = vec![0.; n * s];
        let mut di = vec![0.; n * n * s];
        let mut dq = vec![0.; n * n * s];
        for sample in 0..s {
            let range = sample * n..(sample + 1) * n;
            let matrix = sample * n * n..(sample + 1) * n * n;
            nonlinear.evaluate(
                &state[range.clone()],
                &mut currents[range.clone()],
                &mut charges[range],
                &mut di[matrix.clone()],
                &mut dq[matrix],
            )?;
        }
        if currents
            .iter()
            .chain(&charges)
            .chain(&di)
            .chain(&dq)
            .any(|x| !x.is_finite())
        {
            return Err(numerical(
                "nonfinite physical current, charge or derivative",
            ));
        }
        let mut ispectrum = vec![Complex::ZERO; n * m];
        let mut qspectrum = vec![Complex::ZERO; n * m];
        for row in 0..n {
            let i: Vec<_> = currents.chunks_exact(n).map(|point| point[row]).collect();
            let q: Vec<_> = charges.chunks_exact(n).map(|point| point[row]).collect();
            let ih = self.plan.project(&i)?;
            let qh = self.plan.project(&q)?;
            for mode in 0..m {
                ispectrum[mode * n + row] = ih[mode];
                qspectrum[mode * n + row] = qh[mode];
            }
        }
        let residual = self.residual_spectrum(&spectrum, &ispectrum, &qspectrum);
        let encoded = coords.encode(&residual);
        if encoded.iter().any(|value| !value.is_finite()) {
            return Err(numerical("nonfinite coupled harmonic KCL residual"));
        }
        Ok((
            encoded,
            Sampled {
                di,
                dq,
                ispectrum,
                qspectrum,
                calls: s,
            },
        ))
    }

    fn residual_spectrum(&self, x: &[Complex], i: &[Complex], q: &[Complex]) -> Vec<Complex> {
        let n = self.unknowns;
        let mut residual = vec![Complex::ZERO; x.len()];
        for (h, mode) in self.plan.modes().iter().enumerate() {
            let omega = TAU * mode.frequency_hz;
            for row in 0..n {
                let mut gsum = Complex::ZERO;
                let mut csum = Complex::ZERO;
                for col in 0..n {
                    let value = x[h * n + col];
                    let g = self.g[row * n + col];
                    let c = self.c[row * n + col];
                    gsum = gsum + value * Complex::real(g);
                    csum = csum + value * Complex::real(c);
                }
                let entry = h * n + row;
                residual[entry] = gsum + i[entry] + (csum + q[entry]) * Complex::imaginary(omega)
                    - self.forcing[entry];
            }
        }
        residual
    }

    fn jacobian_action(
        &self,
        coords: &Coordinates,
        sampled: &Sampled,
        dx: &[Real],
    ) -> SpiceResult<Vec<Real>> {
        let full = coords.decode(&self.plan, dx);
        let n = self.unknowns;
        let m = self.plan.modes().len();
        let s = self.plan.sample_count();
        let mut samples = vec![0.; s * n];
        for row in 0..n {
            let spectrum: Vec<_> = full.chunks_exact(n).map(|mode| mode[row]).collect();
            let time = self.plan.synthesize(&spectrum)?;
            for (index, &value) in time.iter().enumerate() {
                samples[index * n + row] = value;
            }
        }
        let mut di = vec![0.; n * s];
        let mut dq = vec![0.; n * s];
        for sample in 0..s {
            for row in 0..n {
                for col in 0..n {
                    let entry = sample * n * n + row * n + col;
                    let value = samples[sample * n + col];
                    di[sample * n + row] += sampled.di[entry] * value;
                    dq[sample * n + row] += sampled.dq[entry] * value;
                }
            }
        }
        let mut ispectrum = vec![Complex::ZERO; n * m];
        let mut qspectrum = vec![Complex::ZERO; n * m];
        for row in 0..n {
            let i: Vec<_> = di.chunks_exact(n).map(|point| point[row]).collect();
            let q: Vec<_> = dq.chunks_exact(n).map(|point| point[row]).collect();
            let ih = self.plan.project(&i)?;
            let qh = self.plan.project(&q)?;
            for mode in 0..m {
                ispectrum[mode * n + row] = ih[mode];
                qspectrum[mode * n + row] = qh[mode];
            }
        }
        let homogeneous = self.residual_spectrum(&full, &ispectrum, &qspectrum);
        let encoded = coords.encode(
            &homogeneous
                .iter()
                .zip(&self.forcing)
                .map(|(&value, &forcing)| value + forcing)
                .collect::<Vec<_>>(),
        );
        if encoded.iter().any(|value| !value.is_finite()) {
            return Err(numerical("nonfinite analytical harmonic Jacobian action"));
        }
        Ok(encoded)
    }

    fn residual_ratio(&self, residual: &[Real], sampled: &Sampled) -> Real {
        // Every residual equation receives a physical scale from its
        // nonlinear model, supplied source and linear MNA, never a shared
        // arbitrary voltage/current numerical norm.
        let coords = Coordinates::new(&self.plan, self.unknowns);
        coords
            .independent
            .iter()
            .zip(residual)
            .map(|(coord, &value)| {
                let index = coord.mode * self.unknowns + coord.row;
                let source = self.forcing[index].magnitude();
                let conduction = sampled.ispectrum[index].magnitude();
                let charge = TAU
                    * self.plan.modes()[coord.mode].frequency_hz.abs()
                    * sampled.qspectrum[index].magnitude();
                let scale = source.max(conduction).max(charge);
                value.abs() / (self.limits.absolute + self.limits.relative * scale)
            })
            .fold(0., Real::max)
    }
}
