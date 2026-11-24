//! Bind the frequency-domain residual to the simulator's prepared physical
//! device loads, using exact state evaluation and independent phase forcing.

use std::sync::Arc;

use crate::devices::functions::{FunctionSpec, SourceFunction};
use crate::devices::{
    AnalysisMode, Circuit, LoadRequest, ModelContext, PreparedRun, StateHistory, TrialState,
    Waveform,
};
use crate::maths::{Pattern, SparseMatrix, Vector};
use crate::primitives::{Real, SpiceError, SpiceResult, filled_vec};

use super::{
    BalanceLimits, BalanceSolution, HarmonicLattice, HarmonicModel, Sample, failure, solve,
};

fn sine(
    lattice: &HarmonicLattice,
    values: &mut [Real],
    [offset, amplitude, frequency, delay, damping, phase]: [Real; 6],
) -> SpiceResult<()> {
    if delay != 0. || damping != 0. || frequency <= 0. {
        return Err(failure(
            "HB needs a periodic SIN with explicit positive frequency, no delay or damping",
        ));
    }
    let index = lattice.locate(frequency).ok_or_else(|| {
        failure(format!(
            "source frequency {frequency} Hz is not in the retained harmonic lattice"
        ))
    })?;
    let angle = phase.rem_euclid(360.).to_radians();
    let (sin, cos) = angle.sin_cos();
    values[0] += offset;
    values[1 + 2 * index] = amplitude.mul_add(sin, values[1 + 2 * index]);
    values[2 + 2 * index] = amplitude.mul_add(cos, values[2 + 2 * index]);
    Ok(())
}

fn waveform(lattice: &HarmonicLattice, wave: &Waveform) -> SpiceResult<Vec<Real>> {
    let mut values = vec![0.; lattice.coordinates()];
    match wave {
        Waveform::Constant(value) => values[0] = *value,
        Waveform::FunctionDefaults(FunctionSpec::Sine(spec)) => sine(
            lattice,
            &mut values,
            [
                spec.offset,
                spec.amplitude,
                spec.frequency.unwrap_or(0.),
                spec.delay.unwrap_or(0.),
                spec.damping.unwrap_or(0.),
                spec.phase.unwrap_or(0.),
            ],
        )?,
        Waveform::Function(function @ SourceFunction::Sine(_)) => {
            let params: Vec<_> = function.parameters().collect();
            let params: [Real; 6] = params
                .try_into()
                .map_err(|_| failure("invalid resolved SIN fields"))?;
            sine(lattice, &mut values, params)?;
        }
        _ => {
            return Err(SpiceError::Unsupported {
                feature: "HB forcing requires a constant or undamped periodic SIN source".into(),
                location: None,
            });
        }
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(failure("nonfinite harmonic forcing"));
    }
    Ok(values)
}

struct CircuitModel<'a> {
    run: PreparedRun<'a>,
    model: ModelContext,
    history: StateHistory,
    trial: Option<TrialState>,
    pattern: Arc<Pattern>,
    linear_charge: SparseMatrix,
    rhs: Vector,
    dc_rhs: Vector,
    forcing: Vec<Real>,
    lattice: &'a HarmonicLattice,
}

impl<'a> CircuitModel<'a> {
    fn new(
        circuit: &'a Circuit,
        model: ModelContext,
        lattice: &'a HarmonicLattice,
        limits: BalanceLimits,
    ) -> SpiceResult<Self> {
        let n = circuit.unknown_count();
        let m = lattice.coordinates();
        let dimension = n
            .checked_mul(m)
            .filter(|count| *count <= limits.unknowns)
            .ok_or_else(|| failure("harmonic circuit unknown limit exceeded"))?;
        if model.source_factor.to_bits() != 1_f64.to_bits() {
            return Err(failure(
                "HB model context must use the physical source factor of one",
            ));
        }
        let run = circuit.prepare_run();
        let linear = run.dae_linear_part(&model)?;
        let dc_rhs = linear.dc_rhs(None)?;
        let mut forcing = filled_vec(dimension, 0., "harmonic source coordinates")?;
        for source in &linear.sources {
            let values = waveform(lattice, &source.waveform)?;
            for &(row, sign) in &source.rows {
                for (value, &coefficient) in forcing[row * m..(row + 1) * m].iter_mut().zip(&values)
                {
                    *value = sign.mul_add(coefficient, *value);
                }
            }
        }
        if forcing.iter().any(|value| !value.is_finite()) {
            return Err(failure("nonfinite aggregate harmonic forcing"));
        }
        let history = circuit.independent_state_history()?;
        let zero = Vector::zeros(n);
        let mut rhs = Vector::zeros(n);
        let mut matrix = circuit.new_matrix()?;
        let mut trial = history.trial();
        run.load(
            &LoadRequest {
                mode: AnalysisMode::OperatingPoint,
                solution: &zero,
                model_context: &model,
                integration: None,
                history: &history,
                forcing: None,
            },
            &mut matrix,
            &mut rhs,
            &mut trial,
        )?;
        let mut charge = Vector::zeros(n);
        let mut dynamic = SparseMatrix::new(n, n);
        run.dae_nonlinear_charge(&model, &zero, &mut charge, &mut dynamic)?;
        let base = circuit
            .matrix_pattern()
            .ok_or_else(|| failure("missing finalized MNA pattern"))?;
        let extra = linear
            .e
            .triplets()
            .iter()
            .chain(dynamic.triplets())
            .map(|term| (term.row, term.col));
        let pattern = Arc::new(base.extended(extra)?);
        let mut linear_charge = SparseMatrix::bound(pattern.clone())?;
        for term in linear.e.triplets() {
            linear_charge.add(term.row, term.col, term.value)?;
        }
        Ok(Self {
            run,
            model,
            history,
            trial: Some(trial),
            pattern,
            linear_charge,
            rhs,
            dc_rhs,
            forcing,
            lattice,
        })
    }
}

impl HarmonicModel for CircuitModel<'_> {
    fn pattern(&self) -> &Arc<Pattern> {
        &self.pattern
    }

    fn evaluate(&mut self, phase: usize, state: &Vector, sample: &mut Sample) -> SpiceResult<()> {
        self.rhs.clear();
        let mut trial = self.history.trial_reusing(self.trial.take());
        let loaded = self.run.load(
            &LoadRequest {
                mode: AnalysisMode::OperatingPoint,
                solution: state,
                model_context: &self.model,
                integration: None,
                history: &self.history,
                forcing: None,
            },
            &mut sample.conductance,
            &mut self.rhs,
            &mut trial,
        );
        self.trial = Some(trial);
        loaded?;
        sample
            .conductance
            .mul_vector_into(state, &mut sample.current)?;
        let m = self.lattice.coordinates();
        let basis = self
            .lattice
            .row(phase)
            .ok_or_else(|| failure("invalid harmonic phase index"))?;
        for (row, output) in sample.current.as_mut_slice().iter_mut().enumerate() {
            let forcing = self.forcing[row * m..(row + 1) * m]
                .iter()
                .zip(basis)
                .fold(0., |sum, (&a, &b)| a.mul_add(b, sum));
            *output =
                (*output - self.rhs.as_slice()[row]) + (self.dc_rhs.as_slice()[row] - forcing);
        }
        self.linear_charge
            .mul_vector_into(state, &mut sample.charge)?;
        for term in self.linear_charge.triplets() {
            sample.capacitance.add(term.row, term.col, term.value)?;
        }
        self.run.dae_nonlinear_charge(
            &self.model,
            state,
            &mut sample.charge,
            &mut sample.capacitance,
        )
    }
}

/// Solve a finalized circuit with explicit independent phase axes.
///
/// This uses the existing exact MNA loads and physical charge laws. A seed
/// is optional; zero Fourier coordinates are used otherwise. Tolerances
/// apply to the physical equations (node currents or branch voltages).
///
/// # Errors
/// Invalid circuit, unsupported physical model/source, invalid seed,
/// exceeded resource limits, or nonlinear nonconvergence.
pub fn solve_circuit(
    circuit: &mut Circuit,
    context: ModelContext,
    lattice: &HarmonicLattice,
    seed: Option<Vector>,
    equation_tolerances: &[Real],
    limits: BalanceLimits,
) -> SpiceResult<BalanceSolution> {
    circuit.finalize()?;
    let mut model = CircuitModel::new(circuit, context, lattice, limits)?;
    let seed = seed.unwrap_or_else(|| Vector::zeros(model.pattern.order() * lattice.coordinates()));
    solve(lattice, &mut model, seed, equation_tolerances, limits)
}
