//! Timed conductance state evolution. The solver owns accepted and tentative
//! history: every transition is a pure calculation from the preceding point.
use crate::primitives::{Real, SpiceError, SpiceResult};

fn invalid(message: &str) -> SpiceError {
    SpiceError::circuit(format!("timed conductance switch: {message}"))
}

/// Validated directional timing and electrical law, shared by model instances.
#[derive(Clone, Copy, Debug)]
pub(super) struct Material {
    conductance: [Real; 2],
    log_conductance: [Real; 2],
    transition: Transition,
    delay: [Real; 2],
    threshold: [Real; 2],
    rising_boundary: RisingBoundary,
    tolerance: Real,
    initial: bool,
}

/// Physical transition law, distinct from propagation delay. Instantaneous
/// inertial changes commit at the deadline, including an exactly coincident
/// reversal; shorter requests cancel. Smooth transitions retain their own
/// established finite-ramp reversal policy.
#[derive(Clone, Copy, Debug)]
pub(super) enum Transition {
    Smooth([Real; 2]),
    InstantInertial,
}

/// Source-selected comparison at the rising threshold. Falling transitions
/// remain strict; the generic finite-slew law retains its original default.
#[derive(Clone, Copy, Debug, Default)]
pub(super) enum RisingBoundary {
    #[default]
    Strict,
    Inclusive,
}

/// Cold parameters, in off/on or falling/rising order.
#[derive(Clone, Copy, Debug)]
pub(super) struct Parameters {
    pub(super) resistance: [Real; 2],
    pub(super) transition: Transition,
    pub(super) delay: [Real; 2],
    pub(super) center: Real,
    pub(super) hysteresis: Real,
    pub(super) rising_boundary: RisingBoundary,
    pub(super) tolerance: Real,
    pub(super) initial: bool,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            resistance: [1e12, 1.],
            transition: Transition::Smooth([1e-9; 2]),
            delay: [0.; 2],
            center: 0.5,
            hysteresis: 1.,
            rising_boundary: RisingBoundary::Strict,
            tolerance: 10e-12,
            initial: false,
        }
    }
}

impl Material {
    pub(super) fn compile(p: Parameters) -> SpiceResult<Self> {
        let conductance = p.resistance.map(Real::recip);
        let threshold = [p.center - p.hysteresis, p.center + p.hysteresis];
        if p.resistance
            .into_iter()
            .chain(conductance)
            .any(|value| !value.is_finite() || value <= 0.)
            || matches!(p.transition, Transition::Smooth(duration) if duration.into_iter().any(|value| !value.is_finite() || value <= 0.))
            || p.delay
                .into_iter()
                .any(|value| !value.is_finite() || value < 0.)
            || !p.hysteresis.is_finite()
            || p.hysteresis < 0.
            || threshold.into_iter().any(|value| !value.is_finite())
            || !p.tolerance.is_finite()
            || p.tolerance < 1e-15
        {
            return Err(invalid(
                "invalid resistance, threshold, delay, duration or crossing tolerance",
            ));
        }
        Ok(Self {
            conductance,
            log_conductance: p.resistance.map(|resistance| -resistance.ln()),
            transition: p.transition,
            delay: p.delay,
            threshold,
            rising_boundary: p.rising_boundary,
            tolerance: p.tolerance,
            initial: p.initial,
        })
    }

    pub(super) const fn instantaneous(self) -> bool {
        matches!(self.transition, Transition::InstantInertial)
    }

    fn receive(self, control: Real, previous: bool) -> bool {
        if previous {
            control >= self.threshold[0]
        } else {
            match self.rising_boundary {
                RisingBoundary::Strict => control > self.threshold[1],
                RisingBoundary::Inclusive => control >= self.threshold[1],
            }
        }
    }

    pub(super) fn equilibrium(self, control: Real) -> SpiceResult<State> {
        if !control.is_finite() {
            return Err(invalid("nonfinite control value"));
        }
        let bit = self.receive(control, self.initial);
        Ok(State {
            time: 0.,
            control,
            anchor_time: 0.,
            anchor_phase: Real::from(u8::from(bit)),
            target: bit,
            requested: bit,
            deadline: None,
        })
    }

    pub(super) fn conductance(self, phase: Real) -> Real {
        if phase <= 0. {
            self.conductance[0]
        } else if phase >= 1. {
            self.conductance[1]
        } else {
            let blend = phase * phase * (-2_f64).mul_add(phase, 3.);
            // Convex combination in log space. Clamp only rounding overshoot
            // against the actual physical rails (never a conductance floor).
            let [off, on] = self.log_conductance;
            let [g0, g1] = self.conductance;
            (1. - blend)
                .mul_add(off, blend * on)
                .exp()
                .clamp(g0.min(g1), g0.max(g1))
        }
    }

    /// Interpolated crossing from a converged candidate, never a Newton
    /// iterate's mutable history. Scaled differences also handle opposite
    /// finite voltages whose direct subtraction would overflow.
    #[expect(
        clippy::suboptimal_flops,
        reason = "crossing interpolation uses separately rounded time coordinates to match the reference event ordering"
    )]
    fn crossing(self, old: State, time: Real, control: Real, request: bool) -> Real {
        let threshold = self.threshold[usize::from(request)];
        let magnitude = old.control.abs().max(control.abs()).max(threshold.abs());
        let delta = control - old.control;
        let fraction = if delta.is_finite() && delta != 0. {
            (threshold - old.control) / delta
        } else if magnitude > 0. {
            (threshold / magnitude - old.control / magnitude)
                / (control / magnitude - old.control / magnitude)
        } else {
            1.
        };
        (old.time + fraction.clamp(0., 1.) * (time - old.time)).clamp(old.time, time)
    }

    #[expect(
        clippy::float_cmp,
        reason = "an already accepted exact threshold point does not require midpoint refinement"
    )]
    pub(super) fn retry_at(self, old: State, candidate: State) -> Option<Real> {
        let time = candidate.time;
        if time <= old.time {
            return None;
        }
        // If a threshold-equality point was already accepted, the crossing
        // may be at old.time. Its newly discovered propagation deadline can
        // nevertheless fall strictly inside this trial. Land on that exact
        // boundary instead of integrating over an unannounced discontinuity.
        let transition = (self.instantaneous()
            && candidate.target != old.target
            && candidate.anchor_time > old.time
            && candidate.anchor_time < time)
            .then_some(candidate.anchor_time);
        if candidate.requested == old.requested {
            return transition;
        }
        let crossing = self.crossing(old, time, candidate.control, candidate.requested);
        // Try the exact estimated crossing before falling back to bracket
        // refinement. Once the remaining bracket is within tolerance, the
        // physical ramp anchor still uses the interpolated crossing.
        let retry = if crossing > old.time && crossing < time {
            Some(crossing)
        } else if crossing <= old.time
            && time - old.time > self.tolerance
            && old.control != self.threshold[usize::from(candidate.requested)]
        {
            let mid = old.time.midpoint(time);
            (mid > old.time && mid < time).then_some(mid)
        } else {
            None
        };
        retry.into_iter().chain(transition).reduce(Real::min)
    }

    pub(super) fn advance(self, old: State, time: Real, control: Real) -> SpiceResult<State> {
        if !time.is_finite() || time < old.time || !control.is_finite() {
            return Err(invalid("invalid candidate time or control"));
        }
        let mut next = old;
        let request = self.receive(control, old.requested);
        if request != old.requested {
            let edge = self.crossing(old, time, control, request);
            if let Some(deadline) = next.deadline.filter(|&deadline| {
                if self.instantaneous() {
                    deadline <= edge
                } else {
                    certainly_before(deadline, edge)
                }
            }) {
                next.start_ramp(self, deadline, next.requested)?;
            }
            next.requested = request;
            if request == next.target {
                next.deadline = None;
            } else if next.moving_at(self, edge)? {
                next.start_ramp(self, edge, request)?;
            } else {
                let delay = self.delay[usize::from(request)];
                if delay == 0. {
                    next.start_ramp(self, edge, request)?;
                } else {
                    next.deadline = Some(future(edge, delay)?);
                }
            }
        }
        if let Some(deadline) = next.deadline.filter(|&deadline| deadline <= time) {
            next.start_ramp(self, deadline, next.requested)?;
        }
        next.time = time;
        next.control = control;
        Ok(next)
    }
}

// A computed crossing and a computed boundary are rounded independently.
// An adjacent representable timestamp is not sufficient evidence that the
// physical crossing followed the boundary. Do not use TIMETOL or the source
// breakpoint merge distance for this ordering decision: those may be much
// larger than the physical separation. Known exact event scheduling is not
// coalesced by this comparison.
fn certainly_before(boundary: Real, crossing: Real) -> bool {
    boundary.next_up() < crossing
}

fn future(time: Real, interval: Real) -> SpiceResult<Real> {
    let end = time + interval;
    if end.is_finite() && (interval == 0. || end > time) {
        Ok(end)
    } else {
        Err(invalid("event time is outside representable range"))
    }
}

/// Accepted state; serializable to seven ordinary finite device slots.
#[derive(Clone, Copy, Debug)]
pub(super) struct State {
    time: Real,
    control: Real,
    anchor_time: Real,
    anchor_phase: Real,
    target: bool,
    requested: bool,
    deadline: Option<Real>,
}

impl State {
    pub(super) const SLOTS: usize = 7;

    fn target_phase(self) -> Real {
        Real::from(u8::from(self.target))
    }

    fn ramp_end(self, material: Material) -> SpiceResult<Real> {
        match material.transition {
            Transition::InstantInertial => Ok(self.anchor_time),
            Transition::Smooth(duration) => {
                let distance = (self.target_phase() - self.anchor_phase).abs();
                future(
                    self.anchor_time,
                    distance * duration[usize::from(self.target)],
                )
            }
        }
    }

    pub(super) fn phase(self, material: Material, time: Real) -> SpiceResult<Real> {
        let Transition::Smooth(duration) = material.transition else {
            return Ok(self.target_phase());
        };
        if time >= self.ramp_end(material)? {
            return Ok(self.target_phase());
        }
        let delta = (time - self.anchor_time).max(0.) / duration[usize::from(self.target)];
        let phase = if self.target {
            self.anchor_phase + delta
        } else {
            self.anchor_phase - delta
        };
        Ok(phase.clamp(0., 1.))
    }

    #[expect(
        clippy::float_cmp,
        reason = "phase rail identity is exact and determines whether the ramp has started"
    )]
    fn moving_at(self, material: Material, time: Real) -> SpiceResult<bool> {
        Ok(!material.instantaneous()
            && self.anchor_phase != self.target_phase()
            && !certainly_before(self.ramp_end(material)?, time))
    }

    fn start_ramp(&mut self, material: Material, time: Real, target: bool) -> SpiceResult<()> {
        self.anchor_phase = self.phase(material, time)?;
        self.anchor_time = time;
        self.target = target;
        if material.instantaneous() {
            self.anchor_phase = self.target_phase();
        }
        self.deadline = None;
        self.ramp_end(material)?;
        Ok(())
    }

    pub(super) fn next_boundary(self, material: Material) -> SpiceResult<Option<Real>> {
        let end = self.ramp_end(material)?;
        Ok(self
            .deadline
            .into_iter()
            .chain([end])
            .filter(|&event| event > self.time)
            .reduce(Real::min))
    }

    pub(super) const fn discrete(self) -> (bool, bool, bool) {
        (self.target, self.requested, self.deadline.is_some())
    }

    pub(super) fn values(self) -> [Real; Self::SLOTS] {
        [
            self.time,
            self.control,
            self.anchor_time,
            self.anchor_phase,
            self.target_phase(),
            Real::from(u8::from(self.requested)),
            self.deadline.unwrap_or(-1.),
        ]
    }

    #[expect(
        clippy::float_cmp,
        reason = "serialized deadline sentinel and Boolean slots are exact discrete encodings"
    )]
    pub(super) fn from_values(values: &[Real]) -> SpiceResult<Self> {
        let &[
            time,
            control,
            anchor_time,
            anchor_phase,
            target,
            request,
            deadline,
        ] = values
        else {
            return Err(invalid("invalid state dimension"));
        };
        if !values.iter().all(|value| value.is_finite())
            || time < 0.
            || anchor_time < 0.
            || anchor_time > time
            || !(0. ..=1.).contains(&anchor_phase)
            || !matches!(target, 0. | 1.)
            || !matches!(request, 0. | 1.)
            || (deadline != -1. && deadline <= time)
        {
            return Err(invalid("invalid accepted switch state"));
        }
        Ok(Self {
            time,
            control,
            anchor_time,
            anchor_phase,
            target: target == 1.,
            requested: request == 1.,
            deadline: (deadline >= 0.).then_some(deadline),
        })
    }

    pub(super) fn at_conductance(self, material: Material) -> SpiceResult<Real> {
        Ok(material.conductance(self.phase(material, self.time)?))
    }

    pub(super) fn discontinuity(self, material: Material, accepted: Self) -> Option<Real> {
        (material.instantaneous() && self.target != accepted.target).then_some(self.anchor_time)
    }

    pub(super) fn step_conductance(
        self,
        material: Material,
        accepted: Option<Self>,
    ) -> SpiceResult<Real> {
        // A step ending at a jump integrates the *left* branch law. The
        // discrete candidate still commits the new state at the deadline,
        // ready for the subsequent interval. Applying the new conductance
        // across the preceding integration interval would inject fictitious
        // charge into capacitive loads (or fictitious flux into inductors).
        if let Some(old) = accepted
            && self.discontinuity(material, old) == Some(self.time)
            && self.time > old.time
        {
            old.at_conductance(material)
        } else {
            self.at_conductance(material)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> SpiceResult<Material> {
        Material::compile(Parameters {
            resistance: [1e6, 1.],
            transition: Transition::Smooth([4., 2.]),
            delay: [3., 2.],
            center: 0.,
            hysteresis: 0.,
            ..Parameters::default()
        })
    }

    #[test]
    fn exact_equal_width_and_reversal_at_endpoint() -> SpiceResult<()> {
        let m = model()?;
        let initial = m.equilibrium(-1.)?;
        let pending = m.advance(initial, 2., 1.)?; // crossing at 1; deadline 3.
        assert_eq!(pending.next_boundary(m)?, Some(3.));
        let equal = m.advance(pending, 4., -1.)?; // crossing at 3 cancels.
        assert_eq!(equal.at_conductance(m)?, 1e-6);
        assert!(equal.next_boundary(m)?.is_none());
        let pending = m.advance(equal, 6., 1.)?; // edge 5, deadline 7
        let moving = m.advance(pending, 8., 1.)?;
        assert_eq!(moving.phase(m, 8.)?, 0.5);
        assert!((moving.at_conductance(m)? - 1e-3).abs() < 1e-15);
        assert_eq!(moving.next_boundary(m)?, Some(9.));
        let reversal = m.advance(moving, 10., -1.)?; // edge at endpoint 9.
        assert_eq!(reversal.phase(m, 10.)?, 0.75);
        assert_eq!(reversal.next_boundary(m)?, Some(13.));
        let complete = m.advance(reversal, 13., -1.)?;
        assert_eq!(complete.at_conductance(m)?, 1e-6);
        assert_eq!(
            State::from_values(&reversal.values())?.values(),
            reversal.values()
        );
        Ok(())
    }

    #[test]
    fn reversal_in_flight_and_rollback_are_pure() -> SpiceResult<()> {
        let m = model()?;
        let initial = m.equilibrium(-1.)?;
        let pending = m.advance(initial, 2., 1.)?;
        let moving = m.advance(pending, 3.5, 1.)?;
        let reversed = m.advance(moving, 4.5, -1.)?; // edge 4, phase .5
        assert_eq!(reversed.phase(m, 4.5)?, 0.375);
        assert_eq!(reversed.next_boundary(m)?, Some(6.));
        let retry = m.advance(moving, 4.5, 1.)?;
        assert_eq!(retry.phase(m, 4.5)?, 0.75);
        assert_eq!(retry.next_boundary(m)?, Some(5.));
        Ok(())
    }

    #[test]
    fn extreme_rails_and_invalid_parameters() -> SpiceResult<()> {
        let m = Material::compile(Parameters {
            resistance: [1e300, 1e-300],
            ..Parameters::default()
        })?;
        assert_eq!(m.conductance(0.), 1e300_f64.recip());
        assert_eq!(m.conductance(1.), 1e-300_f64.recip());
        assert!((m.conductance(0.5) - 1.).abs() < 1e-12);
        assert!(
            Material::compile(Parameters {
                resistance: [0., 1.],
                ..Parameters::default()
            })
            .is_err()
        );
        assert!(
            Material::compile(Parameters {
                delay: [-1., 0.],
                ..Parameters::default()
            })
            .is_err()
        );
        assert!(
            Material::compile(Parameters {
                transition: Transition::Smooth([0., 1.]),
                ..Parameters::default()
            })
            .is_err()
        );
        assert!(
            Material::compile(Parameters {
                tolerance: 0.,
                ..Parameters::default()
            })
            .is_err()
        );
        Ok(())
    }

    fn waveform(name: &str) -> Vec<(Real, Real)> {
        if let Some(width) = name.strip_prefix('w') {
            let end = (1000. + width.parse::<Real>().unwrap()) * 1e-9;
            vec![
                (0., 0.),
                (1e-6, 0.),
                (1.01e-6, 5.),
                (end, 5.),
                (end + 10e-9, 0.),
                (6e-6, 0.),
            ]
        } else if name == "reverse" {
            vec![
                (0., 5.),
                (1e-6, 5.),
                (1.01e-6, 0.),
                (1.35e-6, 0.),
                (1.36e-6, 5.),
                (6e-6, 5.),
            ]
        } else {
            vec![
                (0., 0.),
                (1e-6, 0.),
                (1.01e-6, 5.),
                (1.2e-6, 5.),
                (1.21e-6, 0.),
                (3e-6, 0.),
                (3.01e-6, 5.),
                (4e-6, 5.),
                (4.01e-6, 0.),
                (6e-6, 0.),
            ]
        }
    }

    fn trace_point(
        material: Material,
        points: &[(Real, Real)],
        sample: Real,
        max_step: Real,
    ) -> SpiceResult<Real> {
        let mut state = material.equilibrium(points[0].1)?;
        for pair in points.windows(2) {
            let [(t0, v0), (t1, v1)] = pair else {
                return Err(invalid("invalid waveform segment"));
            };
            let end = sample.min(*t1);
            while state.time < end {
                let mut time = end.min(state.time + max_step);
                if let Some(boundary) = state.next_boundary(material)? {
                    time = time.min(boundary);
                }
                // Retry a crossing at the estimated timestamp, but permit
                // no zero-length step or unrepresentable subdivision.
                let next = loop {
                    let voltage = v0 + (v1 - v0) * (time - t0) / (t1 - t0);
                    let candidate = material.advance(state, time, voltage)?;
                    match material.retry_at(state, candidate) {
                        Some(crossing) if crossing > state.time && crossing.next_up() < time => {
                            time = crossing;
                        }
                        _ => break candidate,
                    }
                };
                state = next;
            }
            if state.time == sample {
                break;
            }
        }
        state.at_conductance(material)
    }

    #[test]
    fn exact_pulse_geometry_matches_reference_samples() -> SpiceResult<()> {
        // Use the physical source corners, but also subdivide each segment
        // and land on internal boundaries. Reference values were obtained
        // independently from the C runtime rather than this interpolation.
        for row in include_str!("../../../tests/fixtures/native-timed-switch/pulse-samples.csv")
            .lines()
            .skip(1)
        {
            let fields: Vec<_> = row.split(',').collect();
            let name = fields[0];
            let time = fields[1].parse::<Real>().unwrap();
            let expected = fields[2].parse::<Real>().unwrap();
            let tolerance = match name {
                "b0" => 1e-15,
                "b1" => 1e-12,
                "b3" => 100e-12,
                "b4" => 1e-9,
                "b5" => 10e-9,
                "b6" => 100e-9,
                _ => 10e-12,
            };
            let material = Material::compile(Parameters {
                resistance: [1e6, 10.],
                transition: Transition::Smooth([200e-9, 100e-9]),
                delay: [300e-9, 500e-9],
                center: 2.5,
                hysteresis: 0.5,
                tolerance,
                ..Parameters::default()
            })?;
            let points = waveform(name);
            for max_step in [Real::INFINITY, 13e-9, 1e-9, 0.25e-9] {
                let actual = trace_point(material, &points, time, max_step)?;
                assert!(
                    (actual - expected).abs() <= expected.abs() * 0.002 + 1e-12,
                    "{name} at {time:e}, step {max_step:e}: {actual:e} vs {expected:e}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn instantaneous_scheduler_matches_independent_inertial_events() -> SpiceResult<()> {
        // Exact binary-fraction pulse times avoid ambiguous decimal rounding
        // at the equality boundary. Reference is C ngspice's d_source and
        // inertial d_buffer event trace, not the finite-slew code model.
        let material = Material::compile(Parameters {
            resistance: [1e6, 10.],
            transition: Transition::InstantInertial,
            delay: [2_f64.powi(-21), 2_f64.powi(-20)],
            center: 0.,
            hysteresis: 0.,
            ..Parameters::default()
        })?;
        let mut state = material.equilibrium(-1.)?;
        for row in include_str!("../../../tests/fixtures/native-timed-switch/instant-inertial.csv")
            .lines()
            .skip(1)
        {
            let fields: Vec<_> = row.split(',').collect();
            let time = fields[0].parse::<Real>().unwrap();
            let control = if fields[1] == "1" { 1. } else { -1. };
            let expected = fields[2] == "1";
            // Approach the edge with the old receiver value, then present
            // the new event at the same timestamp as the digital oracle.
            let left = material.advance(state, time, state.control)?;
            let candidate = material.advance(left, time, control)?;
            assert_eq!(candidate.target, expected, "{row}");
            assert_eq!(
                candidate.at_conductance(material)?,
                material.conductance(Real::from(u8::from(expected)))
            );
            assert_eq!(
                State::from_values(&candidate.values())?.values(),
                candidate.values()
            );
            state = candidate;
        }
        Ok(())
    }

    #[test]
    fn instantaneous_threshold_equality_signed_rails_and_deadline_retry() -> SpiceResult<()> {
        let material = Material::compile(Parameters {
            transition: Transition::InstantInertial,
            resistance: [100., 1e-3],
            delay: [0., 2.],
            center: -3.,
            hysteresis: 1.,
            ..Parameters::default()
        })?;
        let initial = material.equilibrium(-5.)?;
        let equality = material.advance(initial, 3., -2.)?;
        assert_eq!(equality.discrete(), (false, false, false));
        let candidate = material.advance(equality, 7., -1.)?;
        assert_eq!(material.retry_at(equality, candidate), Some(5.));
        let landed = material.advance(equality, 5., -1.5)?;
        assert_eq!(landed.at_conductance(material)?, 1e3);
        assert_eq!(landed.step_conductance(material, Some(equality))?, 0.01);
        assert_eq!(landed.discontinuity(material, equality), Some(5.));
        let next = material.advance(landed, 5.5, -1.5)?;
        assert_eq!(next.step_conductance(material, Some(landed))?, 1e3);
        let lower_equality = material.advance(landed, 6., -4.)?;
        assert_eq!(lower_equality.at_conductance(material)?, 1e3);
        let off = material.advance(lower_equality, 7., -5.)?;
        assert_eq!(off.at_conductance(material)?, 0.01);
        // A rejected recrossing has no effect on a fresh trial of the same
        // accepted state. No artificial ramp endpoint is queued.
        let retry = material.advance(lower_equality, 7., -3.)?;
        assert_eq!(retry.at_conductance(material)?, 1e3);
        assert_eq!(retry.next_boundary(material)?, None);
        Ok(())
    }

    #[test]
    fn inclusive_rising_boundary_preserves_falling_equality_and_delays() -> SpiceResult<()> {
        let material = Material::compile(Parameters {
            transition: Transition::InstantInertial,
            rising_boundary: RisingBoundary::Inclusive,
            resistance: [100., 1e-3],
            delay: [0., 2.],
            center: -3.,
            hysteresis: 1.,
            ..Parameters::default()
        })?;
        assert_eq!(material.equilibrium(-2.)?.at_conductance(material)?, 1e3);
        let initial = material.equilibrium(-5.)?;
        let pending = material.advance(initial, 3., -2.)?;
        assert_eq!(pending.next_boundary(material)?, Some(5.));
        assert_eq!(pending.at_conductance(material)?, 0.01);
        let landed = material.advance(pending, 5., -2.)?;
        assert_eq!(landed.at_conductance(material)?, 1e3);
        let equality = material.advance(landed, 6., -4.)?;
        assert_eq!(equality.at_conductance(material)?, 1e3);
        let off = material.advance(equality, 7., -5.)?;
        assert_eq!(off.at_conductance(material)?, 0.01);
        let zero = Material::compile(Parameters {
            transition: Transition::InstantInertial,
            rising_boundary: RisingBoundary::Inclusive,
            center: 0.,
            hysteresis: 0.,
            ..Parameters::default()
        })?;
        let at = zero.advance(zero.equilibrium(-1.)?, 1., 0.)?;
        assert!(at.target);
        assert!(zero.advance(at, 2., 0.)?.target);
        Ok(())
    }

    #[test]
    fn crossing_retry_preserves_original_physical_anchor() -> SpiceResult<()> {
        let m = model()?;
        let initial = m.equilibrium(-1.)?;
        let candidate = m.advance(initial, 2., 1.)?;
        assert_eq!(m.retry_at(initial, candidate), Some(1.));
        let equality = m.advance(initial, 1., 0.)?;
        assert_eq!(equality.at_conductance(m)?, 1e-6);
        let retry = m.advance(equality, 2., 1.)?;
        assert_eq!(retry.next_boundary(m)?, candidate.next_boundary(m)?);
        assert!(m.retry_at(equality, retry).is_none());
        Ok(())
    }
}
