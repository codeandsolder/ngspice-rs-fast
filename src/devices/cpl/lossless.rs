//! Exact coupled LC travelling-wave transport using accepted delay history.
//!
//! The reciprocal power-normalized transforms preserve V·I and make each
//! mode a pair of lossless scalar delay channels. This is not a spatial
//! ladder: wavefront times are the physical LC eigenvalues, independent of
//! the integrator step or any spatial discretization parameter.
use super::{CoupledLine, invalid, plan::CoupledLinePlan};
use crate::devices::delay::{
    DelayContext, DelayHistory, DelayLine, DelayUpdate, WaveformHistory, WaveformUpdate,
};
use crate::devices::traits::{MnaUnknowns, StampContext};
use crate::primitives::{Complex, Real, SpiceResult};

#[derive(Debug, Clone)]
pub(super) struct ModalBasis {
    pub(super) delays: Vec<Real>,
    // Physical conductor × mode; modal voltage = current_basis^T V,
    // modal current = voltage_basis^T I.
    pub(super) voltage_basis: Vec<Real>,
    pub(super) current_basis: Vec<Real>,
    pub(super) longest: Real,
    pub(super) shortest: Real,
}

impl ModalBasis {
    pub(super) fn prepare(plan: &CoupledLinePlan) -> SpiceResult<Option<Self>> {
        // A semidefinite L or C has ideal instantaneous constraints and does
        // not possess this complete positive-speed travelling-wave basis.
        for material in [&plan.l, &plan.c] {
            if material
                .symmetric_eigenvalues()?
                .first()
                .is_some_and(|&v| v <= 0.)
            {
                return Ok(None);
            }
        }
        let mut modes = plan.exact_lc_modes()?;
        for (index, (v, i)) in modes
            .voltage_basis
            .iter_mut()
            .zip(&mut modes.current_basis)
            .enumerate()
        {
            let root_y = modes.characteristic_admittances[index % modes.conductors].sqrt();
            *v /= root_y;
            *i *= root_y;
            if !v.is_finite() || !i.is_finite() {
                return Err(invalid("nonfinite power-normalized modal basis"));
            }
        }
        let longest = modes.total_delays.iter().copied().fold(0., Real::max);
        let shortest = modes
            .total_delays
            .iter()
            .copied()
            .fold(Real::INFINITY, Real::min);
        Ok(Some(Self {
            delays: modes.total_delays,
            voltage_basis: modes.voltage_basis,
            current_basis: modes.current_basis,
            longest,
            shortest,
        }))
    }

    fn frequency_factors(&self, hz: Real) -> SpiceResult<Vec<Complex>> {
        if !hz.is_finite() || hz < 0. {
            return Err(invalid("invalid modal frequency"));
        }
        self.delays
            .iter()
            .map(|&delay| {
                let angle = -std::f64::consts::TAU * (hz * delay);
                if !angle.is_finite() {
                    return Err(invalid("modal phase overflow"));
                }
                let (im, re) = angle.sin_cos();
                Ok(Complex { re, im })
            })
            .collect()
    }

    pub(super) fn frequency_stamps(
        &self,
        line: &CoupledLine,
        hz: Real,
        unknowns: &MnaUnknowns,
        first: usize,
    ) -> SpiceResult<Vec<(usize, usize, Complex)>> {
        self.stamps(line, &self.frequency_factors(hz)?, unknowns, first)
    }

    // At each port: modal voltage - modal current = delayed incident wave
    // from the opposite end. Branch unknowns are physical inward currents.
    fn stamps(
        &self,
        line: &CoupledLine,
        factors: &[Complex],
        unknowns: &MnaUnknowns,
        first: usize,
    ) -> SpiceResult<Vec<(usize, usize, Complex)>> {
        let n = self.delays.len();
        let mut entries = line.terminal_kcl(unknowns, first)?;
        let mut add = |row: Option<usize>, col: Option<usize>, value: Complex| {
            if let (Some(row), Some(col)) = (row, col)
                && value != Complex::ZERO
            {
                entries.push((row, col, value));
            }
        };
        for end in 0..2 {
            for (mode, &factor) in factors.iter().enumerate() {
                let row = Some(first + end * n + mode);
                for conductor in 0..n {
                    let index = conductor * n + mode;
                    let v = Complex::real(self.current_basis[index]);
                    let i = Complex::real(self.voltage_basis[index]);
                    for (source, voltage, current) in
                        [(end, v, -i), (1 - end, -factor * v, -factor * i)]
                    {
                        let port = source * n + conductor;
                        let (node, ret) = line.physical_pair(port);
                        add(row, unknowns.checked_node_row(node)?, voltage);
                        add(row, unknowns.checked_node_row(ret)?, -voltage);
                        add(row, Some(first + port), current);
                    }
                }
            }
        }
        if entries.iter().any(|(_, _, v)| !v.is_finite()) {
            return Err(invalid("nonfinite modal line stamp"));
        }
        Ok(entries)
    }

    pub(super) fn physical_waves(
        &self,
        line: &CoupledLine,
        context: &DelayContext<'_>,
        current_sign: Real,
    ) -> SpiceResult<Vec<Real>> {
        let n = self.delays.len();
        // Retain the modal value and an absolute evaluation-error bound for
        // each direction. Incident waves are differences of large physical
        // voltage/current terms at a matched port, so |wave| alone cannot
        // bound their numerical uncertainty.
        let mut values = vec![0.; 4 * n];
        let roundoff = 64.
            * Real::EPSILON
            * Real::from(u32::try_from(n).map_err(|_| invalid("modal order overflow"))?);
        if context.branches.len() != 2 * n {
            return Err(invalid("modal line branch count mismatch"));
        }
        for port in 0..2 * n {
            let (node, ret) = line.physical_pair(port);
            let positive = context.node_voltage(node);
            let negative = context.node_voltage(ret);
            let voltage = positive - negative;
            let voltage_uncertainty = roundoff.mul_add(positive.abs(), roundoff * negative.abs());
            let current = context
                .solution
                .get(context.branches.start + port)
                .ok_or_else(|| invalid("missing modal branch current"))?;
            for mode in 0..n {
                let index = (port % n) * n + mode;
                let sample = (port / n) * n + mode;
                values[sample] = self.current_basis[index].mul_add(
                    voltage,
                    self.voltage_basis[index].mul_add(current_sign * current, values[sample]),
                );
                let error_index = 2 * n + sample;
                values[error_index] = self.current_basis[index].abs().mul_add(
                    voltage_uncertainty,
                    self.voltage_basis[index]
                        .abs()
                        .mul_add(roundoff * current.abs(), values[error_index]),
                );
            }
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err(invalid("nonfinite incident modal wave"));
        }
        Ok(values)
    }

    pub(super) fn stamp_transient(
        &self,
        line: &CoupledLine,
        context: &mut StampContext<'_>,
        time: Real,
    ) -> SpiceResult<()> {
        let history = context
            .states
            .delay_history()
            .ok_or_else(|| invalid("missing modal delay history"))?
            .as_waveform()?;
        let n = self.delays.len();
        let mut factors = Vec::with_capacity(n);
        let mut rhs = vec![0.; 2 * n];
        for (mode, &delay) in self.delays.iter().enumerate() {
            let target = time - delay;
            let (left, right, fraction, trial) = interpolation(history, target, time)?;
            factors.push(Complex::real(if trial { fraction } else { 0. }));
            for end in 0..2 {
                let source = (1 - end) * n + mode;
                let a = history
                    .value(left, source)
                    .ok_or_else(|| invalid("missing modal history sample"))?;
                let b = if trial {
                    0.
                } else {
                    history
                        .value(right, source)
                        .ok_or_else(|| invalid("missing modal history sample"))?
                };
                rhs[end * n + mode] = (b - a).mul_add(fraction, a);
            }
        }
        let first = context.branch(0)?;
        for (row, col, value) in self.stamps(line, &factors, context.unknowns, first)? {
            context.matrix.add(row, col, value.re)?;
        }
        for (index, value) in rhs.into_iter().enumerate() {
            context.rhs.add_to(first + index, value)?;
        }
        Ok(())
    }
}

// The retarded time is always strictly earlier than the trial endpoint. If
// it lies in the current step, interpolate *affinely* in the unknown current
// incident wave, not an extrapolated accepted value. Newton thus gets the
// exact Jacobian of this piecewise-linear delay discretization.
fn interpolation(
    history: &WaveformHistory,
    target: Real,
    trial_time: Real,
) -> SpiceResult<(usize, usize, Real, bool)> {
    let len = history.len();
    let first = history
        .time(0)
        .ok_or_else(|| invalid("empty modal delay history"))?;
    let last = history
        .last_time()
        .ok_or_else(|| invalid("empty modal delay history"))?;
    if target < first {
        return Err(invalid("modal delay precedes retained history"));
    }
    if target <= first {
        return Ok((0, 0, 0., false));
    }
    if target > last {
        let fraction = (target - last) / (trial_time - last);
        if !(0. ..=1.).contains(&fraction) || trial_time <= last {
            return Err(invalid("invalid in-step modal delay"));
        }
        return Ok((len - 1, len - 1, fraction, true));
    }
    let mut lo = 0;
    let mut hi = len - 1;
    while hi - lo > 1 {
        let mid = lo.midpoint(hi);
        if history.time(mid).is_some_and(|t| t < target) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let a = history
        .time(lo)
        .ok_or_else(|| invalid("missing modal knot"))?;
    let b = history
        .time(hi)
        .ok_or_else(|| invalid("missing modal knot"))?;
    Ok((lo, hi, (target - a) / (b - a), false))
}

impl DelayLine for CoupledLine {
    fn new_history(&self) -> DelayHistory {
        match &self.propagation {
            super::Propagation::Lossy(plan) => DelayHistory::Cpl(Box::new(plan.new_history())),
            _ => DelayHistory::waveform(4 * self.conductors()),
        }
    }

    fn start(&self, context: &DelayContext<'_>) -> SpiceResult<DelayUpdate> {
        let modes = match &self.propagation {
            super::Propagation::Lossless(modes) => modes,
            super::Propagation::Lossy(plan) => {
                return plan.start(self, context).map(DelayUpdate::Cpl);
            }
            super::Propagation::Spatial => return Err(invalid("missing modal transport plan")),
        };
        let values = if context.initial_conditions {
            vec![0.; 4 * self.conductors()]
        } else {
            modes.physical_waves(self, context, 1.)?
        };
        Ok(DelayUpdate::Waveform(WaveformUpdate {
            reset: Some(vec![(-modes.longest, values.clone()), (0., values)]),
            ..WaveformUpdate::default()
        }))
    }

    fn accept(
        &self,
        history: &DelayHistory,
        context: &DelayContext<'_>,
    ) -> SpiceResult<DelayUpdate> {
        let modes = match &self.propagation {
            super::Propagation::Lossless(modes) => modes,
            super::Propagation::Lossy(plan) => {
                return plan
                    .accept(self, history.as_cpl()?, context)
                    .map(DelayUpdate::Cpl);
            }
            super::Propagation::Spatial => return Err(invalid("missing modal transport plan")),
        };
        let history = history.as_waveform()?;
        let last = history
            .last_time()
            .ok_or_else(|| invalid("empty modal delay history"))?;
        let values = modes.physical_waves(self, context, 1.)?;
        let mut update = WaveformUpdate::default();
        // Retain the last sample before the oldest required retarded time.
        while update.drop_front + 1 < history.len()
            && history
                .time(update.drop_front + 1)
                .is_some_and(|t| t < context.time - modes.longest)
        {
            update.drop_front += 1;
        }
        if context.time <= last {
            return Err(invalid("modal accept time is not increasing"));
        }
        if history.len() >= 2 {
            let prev = history
                .time(history.len() - 2)
                .ok_or_else(|| invalid("missing previous modal knot"))?;
            let width = 2 * self.conductors();
            for (port, &value) in values[..width].iter().enumerate() {
                let a = history
                    .value(history.len() - 2, port)
                    .ok_or_else(|| invalid("missing previous modal wave"))?;
                let b = history
                    .value(history.len() - 1, port)
                    .ok_or_else(|| invalid("missing latest modal wave"))?;
                let ratio = (context.time - last) / (last - prev);
                let expected = (b - a).mul_add(ratio, b);
                // A real change of slope, even if it is a small reflected
                // wave on a large DC bias, creates a physical delayed knot.
                // Suppressing it with the circuit's nonlinear reltol smears
                // the next wavefront across the preceding interval. Only
                // ignore the floating-point uncertainty of this collinearity
                // check, including amplification by unequal time steps.
                let error_a = history
                    .value(history.len() - 2, width + port)
                    .ok_or_else(|| invalid("missing previous modal error bound"))?;
                let error_b = history
                    .value(history.len() - 1, width + port)
                    .ok_or_else(|| invalid("missing latest modal error bound"))?;
                // Propagate the two accepted sample uncertainties through
                // linear extrapolation, then include the current sample.
                // The bounds come from the physical summands rather than
                // their potentially almost-cancelled modal result.
                let roundoff = error_a.mul_add(
                    ratio.abs(),
                    error_b.mul_add(1. + ratio.abs(), values[width + port]),
                );
                if (value - expected).abs() > roundoff {
                    let when = last + modes.delays[port % self.conductors()];
                    if when > context.time {
                        update.breakpoints.push(when);
                    }
                }
            }
        }
        update.append = Some((context.time, values));
        Ok(DelayUpdate::Waveform(update))
    }

    fn timestep_limit(
        &self,
        _history: &DelayHistory,
        _context: &DelayContext<'_>,
    ) -> SpiceResult<Option<Real>> {
        match &self.propagation {
            super::Propagation::Lossless(modes) => Ok(Some(modes.shortest)),
            super::Propagation::Lossy(plan) => Ok(Some(plan.shortest())),
            super::Propagation::Spatial => Err(invalid("missing modal transport plan")),
        }
    }
}
