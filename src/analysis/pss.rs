//! Autonomous periodic steady-state analysis through the companion integrator.

use crate::analysis::companion::{
    OrbitOutput, PeriodicBoundary, TransientEnd, TransientRun, run_transient,
};
use crate::analysis::linear::{number, unsupported};
use crate::analysis::newton::NewtonOptions;
use crate::analysis::periodic_orbit::{PeriodicOrbit, Shooting, ShootingSolution, failure};
use crate::analysis::plan::{TransientOptions, count};
use crate::analysis::reduction::CompensatedSum;
use crate::analysis::{AnalysisContext, AnalysisPlan, Plot, PointSink, TransientPlan};
use crate::devices::Circuit;
use crate::maths::Vector;
use crate::primitives::{Complex, Real, SpiceResult};

const MAX_UNKNOWN_COUNT: usize = 64;
const MIN_RESOLUTION: u16 = 128;
const MAX_POINTS: u16 = 4096;

#[derive(Debug, Clone, PartialEq)]
pub(super) struct PssPlan {
    period: Real,
    startup: Real,
    node: String,
    points: u16,
    harmonics: u16,
    iterations: usize,
    tolerance: Real,
    transient: TransientPlan,
    max_steps: usize,
}

impl PssPlan {
    pub(super) fn resolve(
        arguments: &[String],
        options: TransientOptions,
        newton: &NewtonOptions,
    ) -> SpiceResult<Self> {
        let [frequency, startup, node, rest @ ..] = arguments else {
            return Err(unsupported(
                ".pss requires fguess stabtime oscnode [points [harmonics [sc_iter [steady_coeff]]]]",
            ));
        };
        if rest.len() > 4 {
            return Err(unsupported("too many .pss positional arguments"));
        }
        let period = number(Some(frequency), ".pss guessed frequency")?.recip();
        let startup = number(Some(startup), ".pss stabilization time")?;
        let points = bounded_count(rest.first(), "points", 16, MAX_POINTS, 256)?;
        let harmonics = bounded_count(rest.get(1), "harmonics", 0, points.div_ceil(2), 0)?;
        let iterations = usize::from(bounded_count(rest.get(2), "sc_iter", 1, 100, 20)?);
        let tolerance = number(
            Some(rest.get(3).map_or("5m", String::as_str)),
            ".pss steady coefficient",
        )?;
        let step = period / Real::from(points.max(MIN_RESOLUTION));
        if !period.is_finite()
            || period <= 0.
            || step <= 0.
            || startup < 0.
            || !(0. < tolerance && tolerance <= 0.1)
        {
            return Err(unsupported(
                "invalid .pss frequency, startup or steady coefficient",
            ));
        }
        Ok(Self {
            period,
            startup,
            node: node.clone(),
            points,
            harmonics,
            iterations,
            tolerance,
            transient: TransientPlan::from_bounds([step, period, 0., step], options, newton),
            max_steps: options.max_steps,
        })
    }

    pub(super) fn grid_points(&self) -> usize {
        usize::from(self.points) + 1 + usize::from(self.harmonics)
    }

    pub(super) const fn plot_count(&self) -> usize {
        if self.harmonics == 0 { 1 } else { 2 }
    }
}

fn bounded_count(
    text: Option<&String>,
    name: &str,
    min: u16,
    max: u16,
    default: u16,
) -> SpiceResult<u16> {
    text.map_or(Ok(default), |text| {
        let value = count(text, name, usize::from(min), usize::from(max))?;
        u16::try_from(value).map_err(|_| unsupported(format!("invalid .pss {name}")))
    })
}

struct Discard;

impl PointSink for Discard {
    fn begin(&mut self, _: &Plot) -> SpiceResult<()> {
        Ok(())
    }
    fn selected_columns(&self) -> Option<&[usize]> {
        Some(&[])
    }
    fn point(&mut self, _: &[Complex]) -> SpiceResult<()> {
        Ok(())
    }
    fn end(&mut self) -> SpiceResult<()> {
        Ok(())
    }
}

/// Only the exploratory period is retained, not the stabilization run or
/// finite-difference probes. Observations cannot influence shooting.
struct PhaseTrace {
    columns: Vec<usize>,
    states: Vec<Vector>,
    times: Vec<Real>,
}

impl PhaseTrace {
    fn new(unknowns: usize) -> Self {
        Self {
            columns: (0..=unknowns).collect(),
            states: Vec::new(),
            times: Vec::new(),
        }
    }

    fn scales(&self, absolute: &[Real], tolerance: Real) -> Vector {
        let mut scales = Vector::from_slice(
            &absolute
                .iter()
                .map(|value| value / tolerance)
                .collect::<Vec<_>>(),
        );
        for state in &self.states {
            for (scale, value) in scales.as_mut_slice().iter_mut().zip(state.as_slice()) {
                *scale = scale.max(value.abs());
            }
        }
        scales
    }

    fn phase_seed(&self, row: usize, minimum_swing: Real) -> SpiceResult<(usize, Vector)> {
        let mut low = Real::INFINITY;
        let mut high = Real::NEG_INFINITY;
        for state in &self.states {
            let value = state.as_slice()[row];
            low = low.min(value);
            high = high.max(value);
        }
        if high - low <= minimum_swing {
            return Err(failure(
                "no detectable oscillation at the selected phase node",
            ));
        }
        let center = low.midpoint(high);
        // Choose an actual accepted state near a central crossing, not an
        // interpolated MNA vector with inconsistent nonlinear charges.
        let crossing = self
            .states
            .windows(2)
            .rposition(|pair| {
                let a = pair[0].as_slice()[row];
                let b = pair[1].as_slice()[row];
                (a < center && b >= center) || (a > center && b <= center)
            })
            .ok_or_else(|| failure("phase trace has no transverse crossing"))?;
        Ok((crossing + 1, self.states[crossing + 1].clone()))
    }
}

impl PointSink for PhaseTrace {
    fn begin(&mut self, _: &Plot) -> SpiceResult<()> {
        Ok(())
    }
    fn selected_columns(&self) -> Option<&[usize]> {
        Some(&self.columns)
    }
    fn point(&mut self, values: &[Complex]) -> SpiceResult<()> {
        if values.len() != self.columns.len() {
            return Err(failure("invalid periodic seed trace width"));
        }
        self.times.push(values[0].re);
        self.states.push(Vector::from_slice(
            &values[1..].iter().map(|value| value.re).collect::<Vec<_>>(),
        ));
        Ok(())
    }
    fn end(&mut self) -> SpiceResult<()> {
        Ok(())
    }
}

struct Runner<'a> {
    circuit: &'a mut Circuit,
    request: &'a AnalysisPlan,
    context: &'a AnalysisContext,
    plan: &'a PssPlan,
    accepted: usize,
    rejected: usize,
    resolution: Real,
}

impl Runner<'_> {
    fn integrate(
        &mut self,
        settings: TransientPlan,
        purpose: TransientRun<'_>,
        sink: &mut dyn PointSink,
    ) -> SpiceResult<TransientEnd> {
        let remaining = self
            .plan
            .max_steps
            .saturating_sub(self.accepted + self.rejected);
        if remaining == 0 {
            return Err(failure("aggregate PSS transient step budget exhausted"));
        }
        let result = run_transient(
            self.circuit,
            self.request,
            self.context,
            settings.with_max_steps(remaining),
            purpose,
            sink,
        )?;
        self.accepted += result.stats.accepted;
        self.rejected += result.stats.rejected;
        Ok(result)
    }

    fn shoot(
        &mut self,
        state: &Vector,
        period: Real,
        points: u16,
        output: OrbitOutput<'_>,
        sink: &mut dyn PointSink,
    ) -> SpiceResult<TransientEnd> {
        self.integrate(
            self.plan
                .transient
                .clone()
                .over_period(period, self.resolution),
            TransientRun::Shooting {
                seed: state,
                points,
                output,
            },
            sink,
        )
    }

    fn startup(&mut self) -> SpiceResult<Vector> {
        let duration = self.plan.startup.max(self.plan.transient.step());
        self.integrate(
            self.plan.transient.clone().over_duration(duration),
            TransientRun::Seed,
            &mut Discard,
        )
        .map(|end| end.solution)
    }
}

fn scaled_distance(start: &Vector, end: &Vector, scales: &Vector) -> Real {
    start
        .as_slice()
        .iter()
        .zip(end.as_slice())
        .zip(scales.as_slice())
        .map(|((&a, &b), &scale)| ((b - a) / scale).abs())
        .fold(0., Real::max)
}

impl PhaseTrace {
    /// Find an earlier crossing of the same section and direction whose
    /// complete MNA state is close. A phase-node zero crossing alone is
    /// insufficient for multi-lobed or relaxation trajectories.
    fn earlier_return(
        &self,
        row: usize,
        index: usize,
        scales: &Vector,
        tolerance: Real,
    ) -> Option<Real> {
        let anchor = self.states.get(index)?;
        let target = anchor.get(row)?;
        let direction = target - self.states.get(index.checked_sub(1)?)?.get(row)?;
        if direction == 0. {
            return None;
        }
        let sign = direction.signum();
        self.states
            .get(..index)?
            .windows(2)
            .zip(self.times.windows(2))
            .rev()
            .find_map(|(pair, times)| {
                let a = pair[0].get(row)?;
                let b = pair[1].get(row)?;
                if (a - target) * sign >= 0. || (b - target) * sign < 0. {
                    return None;
                }
                let fraction = (target - a) / (b - a);
                let state = Vector::from_slice(
                    &pair[0]
                        .as_slice()
                        .iter()
                        .zip(pair[1].as_slice())
                        .map(|(&x, &y)| fraction.mul_add(y - x, x))
                        .collect::<Vec<_>>(),
                );
                let duration = self.times[index] - fraction.mul_add(times[1] - times[0], times[0]);
                (duration > 4. * (self.times[1] - self.times[0])
                    && scaled_distance(anchor, &state, scales) <= tolerance)
                    .then_some(duration)
            })
    }
}

fn charge_residual(circuit: &Circuit, end: &TransientEnd, plan: &PssPlan) -> SpiceResult<Real> {
    let initial = end
        .initial_state
        .as_deref()
        .ok_or_else(|| failure("missing initial charge state"))?;
    let final_state = end
        .history
        .accepted(1)
        .ok_or_else(|| failure("missing final charge state"))?;
    let scales = end
        .charge_scales
        .as_deref()
        .ok_or_else(|| failure("missing charge excursion scales"))?;
    let mut maximum: Real = 0.;
    for (index, device) in circuit.devices().iter().enumerate() {
        let rows = circuit
            .state_rows(index)
            .ok_or_else(|| failure("missing periodic device state rows"))?;
        for local in device.truncation_slots() {
            let slot = rows.start + local;
            let (&a, &b) = initial
                .get(slot)
                .zip(final_state.get(slot))
                .ok_or_else(|| failure("missing periodic charge or flux slot"))?;
            let scale = scales[slot].max(plan.transient.chgtol() / plan.tolerance);
            maximum = maximum.max(((b - a) / scale).abs());
        }
    }
    Ok(maximum)
}

fn harmonic_work_exceeds_limit(width: usize, plan: &PssPlan) -> bool {
    width
        .checked_mul(usize::from(plan.points))
        .and_then(|n| n.checked_mul(usize::from(plan.harmonics)))
        .is_none_or(|n| n > 32_000_000)
}

fn spectrum(time: &Plot, plan: &PssPlan, period: Real) -> SpiceResult<Plot> {
    use super::results::{PlotFlags, Variable};
    let mut variables = time.variables().to_vec();
    variables[0] = Variable::new("frequency", "frequency");
    let mut result = Plot::new(
        "pss2",
        "Frequency Domain Periodic Steady State Analysis",
        PlotFlags::Real,
        variables,
    );
    let width = time.variable_count();
    let points = usize::from(plan.points);
    if time.point_count() != points + 1 || harmonic_work_exceeds_limit(width, plan) {
        return Err(failure("periodic harmonic work limit or invalid time grid"));
    }
    let denominator = Real::from(plan.points);
    for harmonic in 0..plan.harmonics {
        let mut sums = vec![(CompensatedSum::default(), CompensatedSum::default()); width - 1];
        for (sample, row) in (0..plan.points).zip(time.rows()) {
            let phase = u32::from(harmonic) * u32::from(sample) % u32::from(plan.points);
            let (sin, cos) = (std::f64::consts::TAU * Real::from(phase) / denominator).sin_cos();
            for ((s, c), value) in sums.iter_mut().zip(&row[1..]) {
                s.add(value.re * sin);
                c.add(value.re * cos);
            }
        }
        let mut row = Vec::with_capacity(width);
        row.push(Complex::real(Real::from(harmonic) / period));
        row.extend(sums.into_iter().map(|(s, c)| {
            let real = c.total() / denominator;
            let magnitude = if harmonic == 0 {
                real
            } else {
                2. * real.hypot(s.total() / denominator)
            };
            Complex::real(magnitude)
        }));
        result.push_point(&row)?;
    }
    Ok(result)
}

pub(super) fn run(
    circuit: &mut Circuit,
    request: &AnalysisPlan,
    context: &AnalysisContext,
    plan: &PssPlan,
    warm: Option<&PeriodicOrbit>,
    sink: &mut dyn PointSink,
) -> SpiceResult<PeriodicOrbit> {
    circuit.finalize()?;
    let topology = circuit.topology_id()?;
    let n = circuit.unknown_count();
    if !(2..=MAX_UNKNOWN_COUNT).contains(&n) {
        return Err(unsupported(format!(
            ".pss single shooting requires 2..={MAX_UNKNOWN_COUNT} MNA unknowns"
        )));
    }
    let width = n + circuit.projected_output_count() + circuit.observations().len() + 1;
    if plan.harmonics != 0 && harmonic_work_exceeds_limit(width, plan) {
        return Err(failure("periodic harmonic work limit"));
    }
    let phase_row = circuit
        .nodes()
        .get(&plan.node)
        .and_then(|node| circuit.unknowns().node_row(node))
        .ok_or_else(|| {
            unsupported(format!(
                "unknown or ground .pss oscillation node {}",
                plan.node
            ))
        })?;
    let newton = &request.dc_settings().newton;
    let absolute: Vec<_> = super::bias::branch_rows(circuit)
        .into_iter()
        .map(|branch| if branch { newton.abstol } else { newton.vntol })
        .collect();
    let mut runner = Runner {
        circuit,
        request,
        context,
        plan,
        accepted: 0,
        rejected: 0,
        resolution: Real::from(plan.points.max(MIN_RESOLUTION)),
    };
    let (seed, shooting) = seed_orbit(&mut runner, topology, phase_row, &absolute, warm)?;
    let verified = refine_orbit(&mut runner, seed, shooting)?;
    emit_orbit(&mut runner, &verified, sink)?;
    Ok(PeriodicOrbit {
        solution: verified.solution,
        topology,
        phase_row,
        charge_residual: verified.charge_error,
        accepted_steps: runner.accepted,
        rejected_steps: runner.rejected,
    })
}

fn seed_orbit(
    runner: &mut Runner<'_>,
    topology: crate::devices::circuit::TopologyId,
    phase_row: usize,
    absolute: &[Real],
    warm: Option<&PeriodicOrbit>,
) -> SpiceResult<(Vector, Shooting)> {
    let plan = runner.plan;
    let (start, guess) = if let Some(warm) = warm {
        if warm.topology != topology
            || warm.phase_row != phase_row
            || !(0.125..=8.).contains(&(warm.period() / plan.period))
        {
            return Err(failure(
                "warm orbit topology, phase section or period is incompatible",
            ));
        }
        (warm.initial().clone(), warm.period())
    } else {
        (runner.startup()?, plan.period)
    };
    let mut trace = PhaseTrace::new(runner.circuit.unknown_count());
    runner.shoot(
        &start,
        guess,
        plan.points.max(MIN_RESOLUTION),
        OrbitOutput::Probe,
        &mut trace,
    )?;
    let scales = trace.scales(absolute, plan.tolerance);
    let (index, phase_seed) =
        trace.phase_seed(phase_row, 8. * runner.request.dc_settings().newton.vntol)?;
    let (seed, guess) = if warm.is_some() {
        (start, guess)
    } else {
        let estimate = trace
            .earlier_return(phase_row, index, &scales, 0.1)
            .unwrap_or(guess);
        (phase_seed, estimate)
    };
    let shooting = Shooting {
        phase_row,
        scales,
        period_guess: guess,
        tolerance: plan.tolerance * 0.25,
        iterations: plan.iterations,
    };
    Ok((seed, shooting))
}

struct VerifiedOrbit {
    solution: ShootingSolution,
    charge_error: Real,
    boundary: Box<PeriodicBoundary>,
}

fn refine_orbit(
    runner: &mut Runner<'_>,
    seed: Vector,
    mut shooting: Shooting,
) -> SpiceResult<VerifiedOrbit> {
    let plan = runner.plan;
    let mut seed = seed;
    let mut evaluations = 1;
    // A separately integrated, finer one-period map verifies closure and
    // charge consistency rather than accepting a discretization artifact.
    let mut last_error = (0., 0.);
    for _ in 0..3 {
        let mut solution = shooting.solve(seed, &mut |state, period| {
            runner
                .shoot(state, period, plan.points, OrbitOutput::Probe, &mut Discard)
                .map(|end| end.solution)
        })?;
        evaluations += solution.map_evaluations;
        runner.resolution *= 2.;
        let mut verification = PhaseTrace::new(runner.circuit.unknown_count());
        let end = runner.shoot(
            &solution.initial,
            solution.period,
            plan.points,
            OrbitOutput::Verify,
            &mut verification,
        )?;
        evaluations += 1;
        let state_error = scaled_distance(&solution.initial, &end.solution, &shooting.scales);
        let charge_error = charge_residual(runner.circuit, &end, plan)?;
        let (index, phase_seed) = verification.phase_seed(
            shooting.phase_row,
            8. * runner.request.dc_settings().newton.vntol,
        )?;
        if let Some(shorter) = verification.earlier_return(
            shooting.phase_row,
            index,
            &shooting.scales,
            plan.tolerance * 4.,
        ) && shorter < solution.period * 0.8
        {
            seed = phase_seed;
            shooting.period_guess = shorter;
            continue;
        }
        last_error = (state_error, charge_error);
        if state_error > plan.tolerance || charge_error > plan.tolerance {
            seed = solution.initial;
            shooting.period_guess = solution.period;
            continue;
        }
        solution.residual = solution.residual.max(state_error);
        solution.map_evaluations = evaluations + 1;
        let boundary = end
            .boundary
            .ok_or_else(|| failure("missing verified periodic boundary"))?;
        return Ok(VerifiedOrbit {
            solution,
            charge_error,
            boundary,
        });
    }

    Err(failure(format!(
        "periodic closure failed under timestep refinement: state {:e}, charge {:e}",
        last_error.0, last_error.1
    )))
}

fn emit_orbit(
    runner: &mut Runner<'_>,
    orbit: &VerifiedOrbit,
    sink: &mut dyn PointSink,
) -> SpiceResult<()> {
    let plan = runner.plan;
    let solution = &orbit.solution;
    let output = OrbitOutput::Emit(&orbit.boundary);
    let final_end = if plan.harmonics == 0 {
        runner.shoot(
            &solution.initial,
            solution.period,
            plan.points,
            output,
            sink,
        )?
    } else {
        let mut collector = super::results::PlotCollector::default();
        let end = runner.shoot(
            &solution.initial,
            solution.period,
            plan.points,
            output,
            &mut collector,
        )?;
        let time = collector
            .finish(1)?
            .pop()
            .ok_or_else(|| failure("missing periodic time plot"))?;
        let frequency = spectrum(&time, plan, solution.period)?;
        sink.complete(time)?;
        sink.complete(frequency)?;
        end
    };
    runner
        .circuit
        .record_final_transient_state(&final_end.history)
}
