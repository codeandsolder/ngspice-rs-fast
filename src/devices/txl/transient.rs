//! `txlload.c`/`txlacct.c`: recursive convolution over accepted V/I history.
//! Only the newest accepted point needs the recurrence accumulators. The
//! waveform ring stores integer picosecond coordinates and terminal values;
//! preparing or rejecting a trial never mutates either accepted state.
#![expect(
    clippy::suboptimal_flops,
    reason = "SPICE numerical reference parity requires the original non-fused C operation order"
)]

use std::collections::VecDeque;

use super::Txl;
use super::pade::{LossyPade, Propagation, Term};
use crate::devices::delay::{DelayContext, DelayHistory, DelayLine, DelayUpdate};
use crate::primitives::{Complex, Real, SpiceError, SpiceResult};
#[cfg(test)]
mod tests;

fn error(message: &str) -> SpiceError {
    SpiceError::Numerical {
        context: "TXL txlload.c".into(),
        message: message.into(),
    }
}

/// C stores integer picoseconds, truncating toward zero. Check the
/// truncated integer, including the fractional interval above `i32::MAX`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Tick(i32);

impl Tick {
    fn from_seconds(time: Real) -> SpiceResult<Self> {
        let ps = time * 1e12;
        crate::primitives::numeric::truncating_signed_integer::<i32>(ps)
            .map(Self)
            .ok_or_else(|| {
                error("time is outside C's signed picosecond history range (about 2.147 ms)")
            })
    }

    fn accepted(time: Real) -> SpiceResult<Self> {
        if time < 0. {
            return Err(error("negative TXL history time"));
        }
        Self::from_seconds(time)
    }

    fn ps(self) -> Real {
        Real::from(self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct TerminalValues {
    voltage: [Real; 2],
    current: [Real; 2],
}

impl TerminalValues {
    fn map(self, f: impl Fn(Real) -> Real) -> Self {
        Self {
            voltage: self.voltage.map(&f),
            current: self.current.map(f),
        }
    }

    fn interpolate(self, next: Self, fraction: Real) -> Self {
        let interpolate = |a: [Real; 2], b: [Real; 2]| {
            std::array::from_fn(|k| crate::devices::functions::affine_value(a[k], b[k], fraction))
        };
        Self {
            voltage: interpolate(self.voltage, next.voltage),
            current: interpolate(self.current, next.current),
        }
    }

    fn finite(self) -> bool {
        self.voltage
            .into_iter()
            .chain(self.current)
            .all(Real::is_finite)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Sample {
    tick: Tick,
    values: TerminalValues,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum PropagationState<const N: usize, const R: usize> {
    Real([[Real; 2]; N]),
    Conjugate {
        real: [[Real; 2]; R],
        pair: [Complex; 2],
    },
}

fn dc_term(term: Term, voltage: [Real; 2]) -> [Real; 2] {
    voltage.map(|v| -v * term.c / term.x)
}

impl<const N: usize, const R: usize> PropagationState<N, R> {
    fn initial(terms: &Propagation<N, R>, voltage: Option<[Real; 2]>) -> Self {
        let real_state = |term| voltage.map_or([0.; 2], |v| dc_term(term, v));
        match terms {
            Propagation::Real(terms) => Self::Real(terms.map(real_state)),
            Propagation::Conjugate { real, pair } => Self::Conjugate {
                real: real.map(real_state),
                // C initializes the real and imaginary storage slots
                // independently, not using complex division.
                pair: voltage.map_or([Complex::new(0., 0.); 2], |v| {
                    v.map(|v| {
                        Complex::new(
                            -v * pair.residue.re / pair.pole.re,
                            -v * pair.residue.im / pair.pole.im,
                        )
                    })
                }),
            },
        }
    }

    fn finite(&self) -> bool {
        match self {
            Self::Real(values) => values.iter().flatten().all(|v| v.is_finite()),
            Self::Conjugate { real, pair } => {
                real.iter().flatten().all(|v| v.is_finite())
                    && pair.iter().all(|v| v.re.is_finite() && v.im.is_finite())
            }
        }
    }

    /// `right_consts_txl`: trapezoidal recurrence in C storage order.
    fn advance(
        &mut self,
        terms: &Propagation<N, R>,
        before: [Real; 2],
        after: [Real; 2],
        dt: Real,
    ) -> SpiceResult<[Real; 2]> {
        let half = 0.5 * dt;
        let mut sums = [0.; 2];
        let mut advance_real = |terms: &[Term], values: &mut [[Real; 2]]| {
            for (term, value) in terms.iter().zip(values) {
                let exp = (term.x * dt).exp();
                for k in 0..2 {
                    value[k] = value[k] * exp + half * term.c * (before[k] * exp + after[k]);
                    sums[k] += value[k];
                }
            }
        };
        match (self, terms) {
            (Self::Real(values), Propagation::Real(terms)) => advance_real(terms, values),
            (
                Self::Conjugate { real, pair },
                Propagation::Conjugate {
                    real: terms,
                    pair: term,
                },
            ) => {
                advance_real(terms, real);
                let exp = (term.pole.re * dt).exp();
                let exp = Complex::new(
                    exp * (term.pole.im * dt).cos(),
                    exp * (term.pole.im * dt).sin(),
                );
                let residue = Complex::new(half * term.residue.re, half * term.residue.im);
                for k in 0..2 {
                    pair[k] = pair[k] * exp
                        + residue * Complex::new(before[k] * exp.re + after[k], before[k] * exp.im);
                    sums[k] += 2. * pair[k].re;
                }
            }
            _ => return Err(error("incompatible TXL recurrence coefficients")),
        }
        Ok(sums)
    }

    /// The unknown fraction stamped into the matrix is only known at accept.
    fn correct(
        &mut self,
        terms: &Propagation<N, R>,
        values: [Real; 2],
        delta: Real,
        ratio: Real,
    ) -> SpiceResult<()> {
        let correction = |value, residue| delta * 0.5e-12 * ratio * value * residue;
        let update_real = |terms: &[Term], state: &mut [[Real; 2]]| {
            for (term, state) in terms.iter().zip(state) {
                for k in 0..2 {
                    state[k] += correction(values[k], term.c);
                }
            }
        };
        match (self, terms) {
            (Self::Real(state), Propagation::Real(terms)) => update_real(terms, state),
            (
                Self::Conjugate { real, pair },
                Propagation::Conjugate {
                    real: terms,
                    pair: term,
                },
            ) => {
                update_real(terms, real);
                for k in 0..2 {
                    pair[k].re += correction(values[k], term.residue.re);
                    pair[k].im += correction(values[k], term.residue.im);
                }
            }
            _ => return Err(error("incompatible TXL recurrence coefficients")),
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Convolution {
    h1: [[Real; 2]; 3],
    h2: PropagationState<3, 1>,
    h3: PropagationState<6, 4>,
}

impl Convolution {
    fn initial(model: &LossyPade, voltage: [Real; 2]) -> Self {
        Self {
            h1: model.h1.map(|term| dc_term(term, voltage)),
            h2: PropagationState::initial(&model.h2, None),
            h3: PropagationState::initial(&model.h3, Some(voltage)),
        }
    }

    fn finite(&self) -> bool {
        self.h1.iter().flatten().all(|v| v.is_finite()) && self.h2.finite() && self.h3.finite()
    }

    fn advance(
        &mut self,
        model: &LossyPade,
        old: TerminalValues,
        before: TerminalValues,
        after: TerminalValues,
        dt: Real,
    ) -> SpiceResult<[Real; 2]> {
        let mut rhs = [0.; 2];
        for (term, state) in model.h1.iter().zip(&self.h1) {
            let exp = (term.x * dt).exp();
            for k in 0..2 {
                rhs[k] -= state[k] * exp + 0.5 * dt * term.c * exp * old.voltage[k];
            }
        }
        let h3 = self
            .h3
            .advance(&model.h3, before.voltage, after.voltage, dt)?;
        let h2 = self
            .h2
            .advance(&model.h2, before.current, after.current, dt)?;
        rhs[0] += h3[1] + h2[1];
        rhs[1] += h3[0] + h2[0];
        Ok(rhs)
    }

    fn accept(
        &mut self,
        model: &LossyPade,
        old: TerminalValues,
        new: TerminalValues,
        delta: Real,
        dt: Real,
        ratio: Real,
    ) -> SpiceResult<()> {
        // update_cnv_txl deliberately rescales the slope in each iteration.
        let mut slopes: [Real; 2] =
            std::array::from_fn(|k| (new.voltage[k] - old.voltage[k]) / delta);
        for (term, state) in model.h1.iter().zip(&mut self.h1) {
            let exponent = term.x * dt;
            let exp = exponent.exp();
            let increment = exponent.exp_m1();
            let factor = term.c / term.x;
            for k in 0..2 {
                slopes[k] *= factor;
                state[k] = (state[k] - slopes[k] * delta) * exp
                    + increment * (new.voltage[k] * factor + 1e12 * slopes[k] / term.x);
            }
        }
        if ratio > 0. {
            self.h2.correct(&model.h2, new.current, delta, ratio)?;
            self.h3.correct(&model.h3, new.voltage, delta, ratio)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct AcceptedState {
    prehistory: [Real; 2],
    convolution: Option<Convolution>,
    clamped: bool,
}

impl AcceptedState {
    fn finite(&self) -> bool {
        self.prehistory.iter().all(|v| v.is_finite())
            && self.convolution.as_ref().is_none_or(Convolution::finite)
    }
}

/// TXL accepted integer-picosecond waveform and latest recurrence state.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct TxlHistory {
    samples: VecDeque<Sample>,
    state: Option<AcceptedState>,
}

#[derive(Debug, Clone, PartialEq)]
enum Change {
    Reset {
        sample: Sample,
        state: AcceptedState,
    },
    Append {
        drop_front: usize,
        sample: Sample,
        state: AcceptedState,
    },
    Unchanged,
}

/// Prepared TXL history change, published only on transient acceptance.
#[derive(Debug, Clone, PartialEq)]
pub struct TxlUpdate(Change);

impl TxlUpdate {
    pub(in crate::devices) const fn is_reset(&self) -> bool {
        matches!(self.0, Change::Reset { .. })
    }
}

impl TxlHistory {
    /// Number of accepted waveform points.
    #[must_use = "the returned count measures accepted waveform history points"]
    pub const fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether transient initialization has supplied a waveform.
    #[must_use = "the result distinguishes uninitialized TXL history from an accepted waveform"]
    pub const fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    fn latest(&self) -> SpiceResult<(&Sample, &AcceptedState)> {
        self.samples
            .back()
            .zip(self.state.as_ref())
            .ok_or_else(|| error("empty TXL history"))
    }

    pub(in crate::devices) fn check(&self, update: &TxlUpdate) -> SpiceResult<()> {
        let (sample, state) = match &update.0 {
            Change::Reset { sample, state } => (sample, state),
            Change::Append {
                drop_front,
                sample,
                state,
            } => {
                let (last, _) = self.latest()?;
                if *drop_front >= self.len() || sample.tick <= last.tick {
                    return Err(error("invalid TXL history advancement"));
                }
                (sample, state)
            }
            Change::Unchanged => return Ok(()),
        };
        if !sample.values.finite() || !state.finite() {
            return Err(error("nonfinite accepted convolution"));
        }
        Ok(())
    }

    pub(in crate::devices) fn apply_validated(&mut self, update: &TxlUpdate) {
        match update.0 {
            Change::Reset { sample, state } => {
                self.samples.clear();
                self.samples.push_back(sample);
                self.state = Some(state);
            }
            Change::Append {
                drop_front,
                sample,
                state,
            } => {
                self.samples.drain(..drop_front);
                self.samples.push_back(sample);
                self.state = Some(state);
            }
            Change::Unchanged => {}
        }
    }

    pub(in crate::devices) fn clear(&mut self) {
        self.samples.clear();
        self.state = None;
    }

    /// Keep the preceding interpolation point even when the cutoff is
    /// exactly equal to an accepted point.
    fn drop_before(&self, cutoff: Real) -> usize {
        self.samples
            .partition_point(|s| s.tick.ps() < cutoff)
            .saturating_sub(1)
    }

    /// `get_pvs_vi_txl`: integer coordinates are already stored, and
    /// fractional delay coordinates are never rounded.
    fn delayed(
        &self,
        target: Real,
        previous: Tick,
        now: Tick,
    ) -> SpiceResult<(TerminalValues, Real)> {
        let (last, state) = self.latest()?;
        if target <= 0. {
            return Ok((
                TerminalValues {
                    voltage: state.prehistory,
                    current: [0.; 2],
                },
                0.,
            ));
        }
        if target > previous.ps() {
            let ratio = (target - previous.ps()) / (now.ps() - previous.ps());
            return Ok((last.values.map(|v| v * (1. - ratio)), ratio));
        }
        let index = self
            .samples
            .partition_point(|s| s.tick.ps() < target)
            .saturating_sub(1);
        let left = &self.samples[index];
        let Some(right) = self.samples.get(index + 1) else {
            return Ok((left.values, 0.));
        };
        let fraction = (target - left.tick.ps()) / (right.tick.ps() - left.tick.ps());
        Ok((left.values.interpolate(right.values, fraction), 0.))
    }
}

struct Trial {
    convolution: Option<Convolution>,
    rhs: [Real; 2],
    ratio: Real,
}

impl Txl {
    fn trial(
        &self,
        history: &TxlHistory,
        now: Tick,
        previous: Tick,
        dt: Real,
    ) -> SpiceResult<Trial> {
        let (last, accepted) = history.latest()?;
        let previous = previous.max(last.tick);
        let c = &self.coefficients;
        let (before, _) = history.delayed(previous.ps() - c.delay_ps, previous, now)?;
        let (after, ratio) = history.delayed(now.ps() - c.delay_ps, previous, now)?;
        let mut convolution = accepted.convolution;
        let mut rhs = match (&mut convolution, &c.lossy) {
            (Some(state), Some(model)) => state.advance(model, last.values, before, after, dt)?,
            (None, None) => [0.; 2],
            _ => return Err(error("incompatible TXL accepted state")),
        };
        rhs[0] += c.atten3 * after.voltage[1] + c.atten2 * after.current[1];
        rhs[1] += c.atten3 * after.voltage[0] + c.atten2 * after.current[0];
        if !convolution.as_ref().is_none_or(Convolution::finite)
            || !rhs.iter().all(|v| v.is_finite())
        {
            return Err(error("nonfinite convolution"));
        }
        Ok(Trial {
            convolution,
            rhs,
            ratio,
        })
    }

    pub(super) fn transient_stamp(
        &self,
        history: &DelayHistory,
        time: Real,
        dt: Real,
        gmin: Real,
    ) -> SpiceResult<([Real; 14], [Real; 2])> {
        let trial = self.trial(
            history.as_txl()?,
            Tick::accepted(time)?,
            Tick::from_seconds(time - dt)?,
            dt,
        )?;
        let coefficients = &self.coefficients;
        let (h1c, h2c, h3c) = coefficients.lossy.as_ref().map_or((0., 0., 0.), |model| {
            (model.h1c, model.h2.residue_sum(), model.h3.residue_sum())
        });
        let admittance = coefficients.admit + 0.5 * dt * h1c;
        let voltage = trial.ratio * (coefficients.atten3 + 0.5 * dt * h3c);
        let current = trial.ratio * (coefficients.atten2 + 0.5 * dt * h2c);
        let leakage = 0.1 * gmin;
        Ok((
            [
                leakage, leakage, leakage, leakage, admittance, admittance, 1., 1., -1., -1.,
                -voltage, -voltage, -current, -current,
            ],
            trial.rhs,
        ))
    }

    fn values(&self, ctx: &DelayContext<'_>) -> SpiceResult<TerminalValues> {
        let a = ctx.branches.start;
        if ctx.branches.len() < 2 {
            return Err(error("missing branch-current binding"));
        }
        Ok(TerminalValues {
            voltage: self.terminals.map(|node| ctx.node_voltage(node)),
            current: [
                ctx.solution
                    .get(a)
                    .ok_or_else(|| error("missing input current"))?,
                ctx.solution
                    .get(a + 1)
                    .ok_or_else(|| error("missing output current"))?,
            ],
        })
    }
}

impl DelayLine for Txl {
    fn new_history(&self) -> DelayHistory {
        DelayHistory::Txl(Box::default())
    }

    fn start(&self, ctx: &DelayContext<'_>) -> SpiceResult<DelayUpdate> {
        if ctx.initial_conditions {
            return Err(self.unsupported("transient uic (TXL has no initial-condition setters)"));
        }
        let values = self.values(ctx)?;
        let state = AcceptedState {
            prehistory: values.voltage,
            convolution: self
                .coefficients
                .lossy
                .as_ref()
                .map(|model| Convolution::initial(model, values.voltage)),
            clamped: false,
        };
        Ok(DelayUpdate::Txl(TxlUpdate(Change::Reset {
            sample: Sample {
                tick: Tick(0),
                values,
            },
            state,
        })))
    }

    fn accept(&self, history: &DelayHistory, ctx: &DelayContext<'_>) -> SpiceResult<DelayUpdate> {
        let history = history.as_txl()?;
        let (last, accepted) = history.latest()?;
        let now = Tick::accepted(ctx.time)?;
        // C adds V/I records only when the integer history time advances.
        if now <= last.tick {
            return Ok(DelayUpdate::Txl(TxlUpdate(Change::Unchanged)));
        }
        let dt = ctx.steps[0];
        let trial = self.trial(history, now, Tick::from_seconds(ctx.time - dt)?, dt)?;
        let values = self.values(ctx)?;
        let mut state = *accepted;
        state.convolution = trial.convolution;
        if let (Some(convolution), Some(model)) = (&mut state.convolution, &self.coefficients.lossy)
        {
            convolution.accept(
                model,
                last.values,
                values,
                now.ps() - last.tick.ps(),
                dt,
                trial.ratio,
            )?;
        }
        // Clamp persists even after the triggering history gap is pruned.
        state.clamped |= dt * 1e12 > self.coefficients.delay_ps;
        Ok(DelayUpdate::Txl(TxlUpdate(Change::Append {
            drop_front: history.drop_before(now.ps() - self.coefficients.delay_ps),
            sample: Sample { tick: now, values },
            state,
        })))
    }

    /// The first oversized step is accepted in C; its clamp governs only
    /// following steps, even after pruning.
    fn timestep_limit(
        &self,
        history: &DelayHistory,
        _ctx: &DelayContext<'_>,
    ) -> SpiceResult<Option<Real>> {
        if self.coefficients.lossy.is_none() {
            return Ok(None);
        }
        let (_, state) = history.as_txl()?.latest()?;
        Ok(state
            .clamped
            .then_some(0.9 * self.coefficients.delay_ps * 1e-12))
    }
}
