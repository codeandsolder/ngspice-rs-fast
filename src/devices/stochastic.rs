//! Deterministic transient noise and random-waveform processes.
//!
//! Every random draw is indexed by an absolute physical-time knot or an RTS
//! transition ordinal, never by the number of Newton or integration trials.
//! Mutable RTS lookup data is a disposable, bounded, synchronized optimization.

use std::collections::VecDeque;
use std::sync::Mutex;

use crate::devices::Limit;
use crate::primitives::{Real, SpiceError, SpiceResult};

const MAX_INDEX: Real = 9_007_199_254_740_991.;
const MAX_INDEX_INTEGER: u64 = 9_007_199_254_740_991;
const INV_53: Real = 1. / 9_007_199_254_740_992.;
const FLICKER_OCTAVES: usize = 16;
const RTS_CACHE: usize = 32;

fn bad(message: impl Into<String>) -> SpiceError {
    SpiceError::circuit(format!("transient stochastic source: {}", message.into()))
}

/// Transient noise in C source-vector order: NA TS NALPHA NAMP RTSAM RTSCAPT RTSEMT.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransientNoise {
    /// White Gaussian knot standard deviation.
    pub white_rms: Real,
    /// Seconds per white/colored noise sample.
    pub step: Real,
    /// Colored-noise spectral slope, between zero and two.
    pub alpha: Real,
    /// RMS of the independent Gaussian innovations feeding the colored
    /// fractional filter; the output power depends on alpha and bandwidth.
    pub flicker_drive_rms: Real,
    /// Two-state RTS amplitude.
    pub rts_amplitude: Real,
    /// Mean duration in the empty RTS state, seconds.
    pub capture_mean: Real,
    /// Mean duration in the occupied RTS state, seconds.
    pub emission_mean: Real,
}

/// Upstream TRRANDOM distribution type 1..=4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RandomKind {
    /// Uniform on [offset-scale,offset+scale), C `drand()` is centered.
    Uniform,
    /// Normal with mean offset, standard deviation scale.
    Gaussian,
    /// Exponential with mean scale and additive offset.
    Exponential,
    /// Poisson with expected count scale and additive offset.
    Poisson,
}
impl RandomKind {
    /// Check the integral upstream distribution discriminator.
    /// # Errors
    /// Not in the 1..=4 set.
    pub fn from_scalar(x: Real) -> SpiceResult<Self> {
        match x {
            1. => Ok(Self::Uniform),
            2. => Ok(Self::Gaussian),
            3. => Ok(Self::Exponential),
            4. => Ok(Self::Poisson),
            _ => Err(bad("TRRANDOM type must be 1, 2, 3 or 4")),
        }
    }
}

/// Transient held-random parameters in C order: type TS TD P1 P2.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransientRandom {
    /// Which distribution to sample.
    pub kind: RandomKind,
    /// Time between fixed-grid accepted sample transitions.
    pub step: Real,
    /// Time of first random sample (before that, offset is held).
    pub delay: Real,
    /// Distribution scale (Gaussian sigma, exponential/Poisson mean).
    pub scale: Real,
    /// Additive location parameter.
    pub offset: Real,
}

/// Stochastic law, distinct from small-signal spectral noise generators.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StochasticSpec {
    /// White plus finite-band colored noise and optional exact RTS.
    Noise(TransientNoise),
    /// A time-grid-held draw of one of four C distributions.
    Random(TransientRandom),
}
impl StochasticSpec {
    /// Validate physical units and the representable stationary regime.
    /// # Errors
    /// Nonfinite, negative, unsupported, or unresolved time-band inputs.
    pub fn validate(self) -> SpiceResult<()> {
        match self {
            Self::Noise(n) => {
                let fields = [
                    n.white_rms,
                    n.step,
                    n.alpha,
                    n.flicker_drive_rms,
                    n.rts_amplitude,
                    n.capture_mean,
                    n.emission_mean,
                ];
                if fields.iter().any(|v| !v.is_finite())
                    || n.white_rms < 0.
                    || n.flicker_drive_rms < 0.
                    || n.rts_amplitude < 0.
                    || n.step < 0.
                    || !(0. ..2.).contains(&n.alpha)
                {
                    return Err(bad("invalid TRNOISE RMS, spacing or 1/f exponent"));
                }
                if (n.white_rms > 0. || n.flicker_drive_rms > 0.) && n.step == 0. {
                    return Err(bad("white or colored noise needs positive sampling time"));
                }
                if n.flicker_drive_rms > 0. && n.alpha == 0. {
                    return Err(bad("nonzero flicker amplitude needs positive alpha"));
                }
                if n.rts_amplitude > 0. && (n.capture_mean <= 0. || n.emission_mean <= 0.) {
                    return Err(bad("nonzero RTS needs positive capture/emission means"));
                }
            }
            Self::Random(r) => {
                if !(r.step.is_finite()
                    && r.step > 0.
                    && r.delay.is_finite()
                    && r.delay >= 0.
                    && r.scale.is_finite()
                    && r.scale >= 0.
                    && r.offset.is_finite())
                {
                    return Err(bad("invalid TRRANDOM distribution/time parameters"));
                }
                if r.kind == RandomKind::Poisson && r.scale > MAX_INDEX {
                    return Err(bad("Poisson mean not representable exactly in binary64"));
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct Transition {
    index: u64,
    time: Real,
}
#[derive(Debug, Default)]
struct RtsCache {
    recent: VecDeque<Transition>,
}
impl RtsCache {
    fn at_or_before(&self, t: Real, limit: Limit) -> Transition {
        self.recent
            .iter()
            .rev()
            .copied()
            .find(|event| {
                if limit == Limit::Left {
                    event.time < t
                } else {
                    event.time <= t
                }
            })
            .unwrap_or(Transition { index: 0, time: 0. })
    }
    fn remember(&mut self, event: Transition) {
        if self
            .recent
            .back()
            .is_some_and(|old| event.index <= old.index)
        {
            return;
        }
        self.recent.push_back(event);
        if self.recent.len() > RTS_CACHE {
            self.recent.pop_front();
        }
    }
}

/// Source-bound reproducible stochastic waveform. Its cache can be dropped
/// or cloned independently without changing any generated sample or event.
#[derive(Debug)]
pub struct StochasticWaveform {
    spec: StochasticSpec,
    seed: u64,
    identity: u64,
    candidate_identity: u64,
    dc: Real,
    flicker_bands: [Real; FLICKER_OCTAVES],
    cache: Mutex<RtsCache>,
}
impl Clone for StochasticWaveform {
    fn clone(&self) -> Self {
        Self {
            spec: self.spec,
            seed: self.seed,
            identity: self.identity,
            candidate_identity: self.candidate_identity,
            dc: self.dc,
            flicker_bands: self.flicker_bands,
            cache: Mutex::default(),
        }
    }
}
impl PartialEq for StochasticWaveform {
    fn eq(&self, rhs: &Self) -> bool {
        self.spec == rhs.spec
            && self.seed == rhs.seed
            && self.identity == rhs.identity
            && self.candidate_identity == rhs.candidate_identity
            && self.dc == rhs.dc
    }
}
impl StochasticWaveform {
    /// Create a deterministic unbound process.
    /// # Errors
    /// Unsupported source law.
    pub fn new(spec: StochasticSpec) -> SpiceResult<Self> {
        spec.validate()?;
        let flicker_bands = match spec {
            StochasticSpec::Noise(plan) if plan.flicker_drive_rms > 0. => {
                fractional_band_weights(plan.alpha)
            }
            _ => [0.; FLICKER_OCTAVES],
        };
        Ok(Self {
            spec,
            seed: 0x4e47_5350_4943_4552,
            identity: 0,
            candidate_identity: 0,
            dc: 0.,
            flicker_bands,
            cache: Mutex::default(),
        })
    }
    /// Override source stream seed without changing physical parameters.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self.cache = Mutex::default();
        self
    }
    /// Bind a distinct stable canonical source identity.
    #[must_use]
    pub fn with_source_name(mut self, name: &str) -> Self {
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        for byte in name.bytes() {
            hash ^= u64::from(byte.to_ascii_lowercase());
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
        self.identity = hash;
        self.cache = Mutex::default();
        self
    }
    /// Bind one optimizer candidate's stable identity without consuming RNG.
    /// Rebinding A/B/A replays A bitwise, independent of worker scheduling.
    #[must_use]
    pub fn with_candidate_identity(mut self, candidate: u64) -> Self {
        self.candidate_identity = candidate;
        self.cache = Mutex::default();
        self
    }
    /// Explicit DC bias added only to transient source excitation.
    /// # Errors
    /// Nonfinite offset.
    pub fn with_dc(mut self, dc: Real) -> SpiceResult<Self> {
        if !dc.is_finite() {
            return Err(bad("nonfinite DC offset"));
        }
        self.dc = dc;
        Ok(self)
    }
    /// Immutable physical law.
    #[must_use]
    pub const fn spec(&self) -> StochasticSpec {
        self.spec
    }
    /// Source value in an operating point.
    #[must_use]
    pub const fn time_zero(&self) -> Real {
        self.dc
            + if let StochasticSpec::Random(plan) = self.spec {
                plan.offset
            } else {
                0.
            }
    }
    const fn key(&self) -> u64 {
        mix(self.seed ^ self.identity.rotate_left(23) ^ self.candidate_identity.rotate_left(7))
    }
    fn next_transition(&self, previous: Transition) -> SpiceResult<Transition> {
        let StochasticSpec::Noise(n) = self.spec else {
            return Err(bad("random sample has no RTS transitions"));
        };
        let mean = if previous.index % 2 == 0 {
            n.capture_mean
        } else {
            n.emission_mean
        };
        let u = uniform(self.key(), 0x5254_5320, previous.index, 0);
        let next = mean.mul_add(-u.ln(), previous.time);
        if !next.is_finite() || next <= previous.time {
            return Err(bad("RTS event time below floating-point time resolution"));
        }
        Ok(Transition {
            time: next,
            index: previous.index + 1,
        })
    }
    fn rts_state(&self, t: Real, limit: Limit) -> SpiceResult<(bool, Transition)> {
        // This bounded cache is only an accelerator. It is shared through
        // Arc-backed immutable waveforms across concurrent evaluations, so
        // every lookup and advancement must be serialized. An interrupted
        // calculation can never change the physical process: all random
        // values are indexed by source identity and event ordinal.
        let mut cache = self.cache.lock().unwrap_or_else(|poisoned| {
            // A poisoned cache cannot be trusted as a starting point.
            // Rebuilding from the absolute-time origin is exact.
            let mut cache = poisoned.into_inner();
            cache.recent.clear();
            self.cache.clear_poison();
            cache
        });
        let mut current = cache.at_or_before(t, limit);
        loop {
            let next = self.next_transition(current)?;
            let passed = if limit == Limit::Left {
                next.time < t
            } else {
                next.time <= t
            };
            if !passed {
                break;
            }
            current = next;
            cache.remember(current);
        }
        drop(cache);
        Ok((current.index % 2 == 1, current))
    }
    /// Evaluate physical time independently of rejected or accepted trials.
    /// # Errors
    /// Time outside the representable grid or nonfinite noise output.
    pub fn value_at(&self, t: Real, limit: Limit) -> SpiceResult<Real> {
        if !(t.is_finite() && t >= 0.) {
            return Err(bad("invalid physical time"));
        }
        let v = match self.spec {
            StochasticSpec::Noise(n) => {
                let white = if n.white_rms > 0. && t > 0. {
                    n.white_rms * interpolated(self.key(), 0x5748_4954, t, n.step)?
                } else {
                    0.
                };
                let flicker = if n.flicker_drive_rms > 0. && t > 0. {
                    colored(
                        self.key(),
                        t,
                        n.step,
                        n.flicker_drive_rms,
                        &self.flicker_bands,
                    )?
                } else {
                    0.
                };
                let rts = if n.rts_amplitude > 0. && self.rts_state(t, limit)?.0 {
                    n.rts_amplitude
                } else {
                    0.
                };
                (white + flicker) + rts
            }
            StochasticSpec::Random(r) => {
                if (limit == Limit::Left && t <= r.delay) || t < r.delay {
                    r.offset
                } else {
                    let (mut index, fraction) = grid_index((t - r.delay) / r.step)?;
                    if limit == Limit::Left && fraction.to_bits() == 0_f64.to_bits() && index > 0 {
                        index -= 1;
                    }
                    random_sample(self.key(), index, r)?
                }
            }
        } + self.dc;
        if v.is_finite() {
            Ok(v)
        } else {
            Err(bad("noise amplitude overflows binary64"))
        }
    }
    /// Lazily enumerate physical noise knots and RTS transitions.
    /// # Errors
    /// Invalid window/grid.
    pub fn events(&self, start: Real, end: Real) -> SpiceResult<StochasticEvents> {
        if !(start.is_finite() && end.is_finite() && start >= 0. && end >= start) {
            return Err(bad("invalid event window"));
        }
        let grid = match self.spec {
            StochasticSpec::Noise(n) if n.step > 0. => {
                Some(GridEvents::new(n.step, 0., start, end)?)
            }
            StochasticSpec::Random(r) => Some(GridEvents::new(r.step, r.delay, start, end)?),
            StochasticSpec::Noise(_) => None,
        };
        let next_rts = match self.spec {
            StochasticSpec::Noise(n) if n.rts_amplitude > 0. => {
                let (_, previous) = self.rts_state(start, Limit::Left)?;
                Some(self.next_transition(previous)?)
            }
            _ => None,
        };
        Ok(StochasticEvents {
            source: self.clone(),
            grid,
            next_rts,
            start,
            end,
        })
    }
}

const fn mix(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
const fn draw(key: u64, stream: u64, index: u64, ordinal: u64) -> u64 {
    mix(key
        ^ mix(stream)
        ^ mix(index.wrapping_add(0x243f_6a88_85a3_08d3))
        ^ mix(ordinal.wrapping_add(0x1319_8a2e_0370_7344)))
}
/// Convert integer through 32-bit limbs; all binary64 grid integers
/// are exactly represented without relying on a narrowing cast.
fn real_u64(value: u64) -> Real {
    let hi = u32::try_from(value >> 32).unwrap_or_default();
    let lo = u32::try_from(value & u64::from(u32::MAX)).unwrap_or_default();
    4_294_967_296_f64.mul_add(Real::from(hi), Real::from(lo))
}
/// Integer floor of bounded nonnegative IEEE754 binary64, without casting.
const fn floor_nonnegative(value: Real) -> u64 {
    let bits = value.to_bits();
    let exponent = (bits >> 52) & 0x7ff;
    if exponent < 1023 {
        return 0;
    }
    let significand = (bits & ((1_u64 << 52) - 1)) | (1_u64 << 52);
    if exponent >= 1075 {
        significand << (exponent - 1075)
    } else {
        significand >> (1075 - exponent)
    }
}
fn uniform(key: u64, stream: u64, index: u64, ordinal: u64) -> Real {
    (real_u64(draw(key, stream, index, ordinal) >> 11) + 0.5) * INV_53
}
fn gaussian(key: u64, stream: u64, index: u64) -> Real {
    let unit = uniform(key, stream, index, 0);
    let theta = std::f64::consts::TAU * uniform(key, stream, index, 1);
    (-2. * unit.ln()).sqrt() * theta.cos()
}
fn grid_index(value: Real) -> SpiceResult<(u64, Real)> {
    if !value.is_finite() || !(0. ..=MAX_INDEX).contains(&value) {
        return Err(bad(
            "physical time requires more grid bits than binary64 supplies",
        ));
    }
    let index = floor_nonnegative(value);
    Ok((index, value - real_u64(index)))
}
fn interpolated(key: u64, stream: u64, time: Real, step: Real) -> SpiceResult<Real> {
    let (index, fraction) = grid_index(time / step)?;
    let sample = |n| if n == 0 { 0. } else { gaussian(key, stream, n) };
    let left = sample(index);
    let right = sample(index + 1);
    Ok((right - left).mul_add(fraction, left))
}
/// Group the exact C Kasdin fractional-difference kernel's squared
/// coefficients by octave. h[k] = h[k-1] * (alpha/2 + k - 1)/k. This
/// retains NAMP as innovation amplitude, not a chosen output RMS, and
/// makes a bounded-bandwidth stochastic approximation to the C FFT filter.
fn fractional_band_weights(alpha: Real) -> [Real; FLICKER_OCTAVES] {
    let mut weights = [0.; FLICKER_OCTAVES];
    let mut k = 1_u32;
    let mut h = 1.;
    for (octave, weight) in weights.iter_mut().enumerate() {
        let end = 1_u32 << u32::try_from(octave + 1).unwrap_or_default();
        let mut energy = 0.;
        while k < end {
            h *= alpha.mul_add(0.5, Real::from(k - 1)) / Real::from(k);
            energy = h.mul_add(h, energy);
            k += 1;
        }
        // The octave's interpolated independent Gaussian has mean square
        // 2/3 over phase. Scale its innovation to retain that band's
        // physical power from the C coefficient sequence.
        *weight = (1.5 * energy).sqrt();
    }
    weights
}

fn colored(
    key: u64,
    time: Real,
    step: Real,
    innovation_rms: Real,
    weights: &[Real; FLICKER_OCTAVES],
) -> SpiceResult<Real> {
    // The current Kasdin innovation (h0 = 1) is interpolated at the
    // physical sample spacing; higher-lag fractional weights are
    // represented by bounded, independent octave bands.
    let mut sum = interpolated(key, 0x464c_4943, time, step)?;
    for (octave, weight) in weights.iter().copied().enumerate() {
        let spacing = step * 2_f64.powi(i32::try_from(octave).unwrap_or_default());
        let sample = interpolated(
            key,
            0x464c_4943 ^ (u64::try_from(octave).unwrap_or_default() + 1),
            time,
            spacing,
        )?;
        sum = weight.mul_add(sample, sum);
    }
    Ok(sum * innovation_rms)
}

fn log_factorial(value: Real) -> Real {
    if value <= 64. {
        let upper = u32::try_from(floor_nonnegative(value)).unwrap_or_default();
        (2..=upper).map(|index| Real::from(index).ln()).sum()
    } else {
        let inv = value.recip();
        let stirling = (value + 0.5).mul_add(value.ln(), -value);
        stirling + 0.918_938_533_204_672_7 + inv / 12. - inv.powi(3) / 360. + inv.powi(5) / 1260.
    }
}
fn poisson(key: u64, index: u64, lambda: Real) -> Real {
    if lambda == 0. {
        return 0.;
    }
    if lambda < 30. {
        let bound = (-lambda).exp();
        let mut product = 1.;
        let mut attempt = 0_u64;
        loop {
            product *= uniform(key, 0x504f_4953, index, attempt);
            if product <= bound {
                return real_u64(attempt);
            }
            attempt += 1;
        }
    }
    // Hoermann transformed rejection, constant expected draw count at large lambda.
    let shape = 2.53_f64.mul_add(lambda.sqrt(), 0.931);
    let adjustment = 0.02483_f64.mul_add(shape, -0.059);
    let inv_alpha = 1.1239 + 1.1328 / (shape - 3.4);
    let immediate_accept = 0.9277 - 3.6224 / (shape - 2.);
    let mut attempt = 0_u64;
    loop {
        let centered = uniform(key, 0x504f_4953, index, 2 * attempt) - 0.5;
        let draw = uniform(key, 0x504f_4953, index, 2 * attempt + 1);
        attempt += 1;
        let tail = 0.5 - centered.abs();
        let candidate = (2. * adjustment / tail + shape)
            .mul_add(centered, lambda + 0.43)
            .floor();
        if candidate < 0. {
            continue;
        }
        if tail >= 0.07 && draw <= immediate_accept {
            return candidate;
        }
        if tail < 0.013 && draw > tail {
            continue;
        }
        let log_test = draw.ln() + inv_alpha.ln() - (adjustment / (tail * tail) + shape).ln();
        if log_test <= candidate.mul_add(lambda.ln(), -lambda) - log_factorial(candidate) {
            return candidate;
        }
    }
}
fn random_sample(key: u64, index: u64, r: TransientRandom) -> SpiceResult<Real> {
    let v = match r.kind {
        RandomKind::Uniform => r.scale * 2_f64.mul_add(uniform(key, 0x554e_4946, index, 0), -1.),
        RandomKind::Gaussian => r.scale * gaussian(key, 0x4741_5553, index),
        RandomKind::Exponential => -r.scale * uniform(key, 0x4558_504f, index, 0).ln(),
        RandomKind::Poisson => poisson(key, index, r.scale),
    } + r.offset;
    if v.is_finite() {
        Ok(v)
    } else {
        Err(bad("random value overflow"))
    }
}

/// Type of physical source event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StochasticEventKind {
    /// Sample-grid knot or held random step.
    Sample,
    /// Exact RTS transition into the other state.
    Telegraph,
}
/// One lazily generated source event in the requested physical time window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StochasticEvent {
    /// Absolute seconds.
    pub time: Real,
    /// Sample or telegraph transition.
    pub kind: StochasticEventKind,
}
#[derive(Debug, Clone)]
struct GridEvents {
    step: Real,
    delay: Real,
    next: u64,
    end: Real,
}
impl GridEvents {
    fn new(step: Real, delay: Real, start: Real, end: Real) -> SpiceResult<Self> {
        if start > MAX_INDEX.mul_add(step, delay) {
            return Err(bad(
                "source grid cannot represent requested physical window",
            ));
        }
        let index = ((start - delay) / step).max(0.);
        let mut next = grid_index(index)?.0;
        if real_u64(next).mul_add(step, delay) < start {
            next += 1;
        }
        Ok(Self {
            step,
            delay,
            next,
            end,
        })
    }
    fn next_time(&self) -> Option<Real> {
        if self.next > MAX_INDEX_INTEGER {
            return None;
        }
        let t = self.step.mul_add(real_u64(self.next), self.delay);
        (t.is_finite() && t <= self.end).then_some(t)
    }
    const fn advance(&mut self) {
        self.next += 1;
    }
}
/// O(1)-memory iterator that merges sample knots and exact RTS transitions,
/// without ever constructing a full transient waveform vector.
#[derive(Debug, Clone)]
pub struct StochasticEvents {
    source: StochasticWaveform,
    grid: Option<GridEvents>,
    next_rts: Option<Transition>,
    start: Real,
    end: Real,
}
impl StochasticEvents {
    /// Fetch the next event in ascending physical time.
    #[must_use]
    pub fn next_event(&mut self) -> Option<StochasticEvent> {
        loop {
            let grid = self.grid.as_ref().and_then(GridEvents::next_time);
            let rts = self.next_rts.map(|event| event.time);
            let next = [grid, rts].into_iter().flatten().min_by(Real::total_cmp)?;
            if next > self.end {
                return None;
            }
            let sample = grid == Some(next);
            let telegraph = rts == Some(next);
            if sample && let Some(grid) = &mut self.grid {
                grid.advance();
            }
            if telegraph {
                self.next_rts = self
                    .next_rts
                    .and_then(|before| self.source.next_transition(before).ok());
            }
            if next < self.start {
                continue;
            }
            return Some(StochasticEvent {
                time: next,
                kind: if telegraph {
                    StochasticEventKind::Telegraph
                } else {
                    StochasticEventKind::Sample
                },
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn white() -> StochasticWaveform {
        StochasticWaveform::new(StochasticSpec::Noise(TransientNoise {
            white_rms: 1.,
            step: 0.001,
            alpha: 0.,
            flicker_drive_rms: 0.,
            rts_amplitude: 0.,
            capture_mean: 0.,
            emission_mean: 0.,
        }))
        .unwrap()
    }
    #[test]
    fn exact_time_identity_clone_and_independent_sources() {
        let a = white().with_source_name("VONE");
        let b = a.clone();
        let c = white().with_source_name("VTWO");
        for t in [0., 0.001, 0.0025, 1., 22.123] {
            assert_eq!(
                a.value_at(t, Limit::Left).unwrap().to_bits(),
                b.value_at(t, Limit::Left).unwrap().to_bits()
            );
        }
        assert_ne!(
            a.value_at(0.03, Limit::Right).unwrap(),
            c.value_at(0.03, Limit::Right).unwrap()
        );
    }
    #[test]
    fn interpolation_is_physical_time_based() {
        let a = white();
        let x = a.value_at(0.001, Limit::Left).unwrap();
        let y = a.value_at(0.002, Limit::Left).unwrap();
        let z = a.value_at(0.0015, Limit::Right).unwrap();
        assert!((z - (x + y) * 0.5).abs() < 1e-14);
        assert_eq!(a.value_at(0., Limit::Right).unwrap(), 0.);
    }
    #[test]
    fn shared_stochastic_waveforms_are_send_sync_and_replay_across_threads() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<StochasticWaveform>();
        assert_send_sync::<crate::devices::Waveform>();

        let shared = std::sync::Arc::new(
            StochasticWaveform::new(StochasticSpec::Noise(TransientNoise {
                white_rms: 0.25,
                step: 0.001,
                alpha: 0.,
                flicker_drive_rms: 0.,
                rts_amplitude: 2.,
                capture_mean: 0.0009,
                emission_mean: 0.0014,
            }))
            .unwrap()
            .with_seed(0xabc1_2345)
            .with_source_name("V_SHARED"),
        );
        let mut event_source = shared.events(0., 0.1).unwrap();
        let events: Vec<_> = std::iter::from_fn(|| event_source.next_event())
            .filter(|event| event.kind == StochasticEventKind::Telegraph)
            .take(40)
            .collect();
        assert!(events.len() >= 4);
        std::thread::scope(|scope| {
            for worker in 0..8 {
                let shared = std::sync::Arc::clone(&shared);
                let events = &events;
                scope.spawn(move || {
                    // An independent clone resets the cache but not the
                    // absolute-time random stream. Each worker races on
                    // the original source's one shared cache.
                    let reference = shared.as_ref().clone();
                    for step in 0..500 {
                        let index = (step * 97 + worker * 43) % 500;
                        let t = index as Real * 0.0002;
                        let limit = if step % 2 == 0 {
                            Limit::Left
                        } else {
                            Limit::Right
                        };
                        let got = shared.value_at(t, limit).unwrap();
                        let expected = reference.value_at(t, limit).unwrap();
                        assert_eq!(got.to_bits(), expected.to_bits());
                    }
                    for event in events {
                        for limit in [Limit::Left, Limit::Right] {
                            let got = shared.value_at(event.time, limit).unwrap();
                            let expected = reference.value_at(event.time, limit).unwrap();
                            assert_eq!(got.to_bits(), expected.to_bits());
                        }
                    }
                });
            }
        });
    }

    #[test]
    fn accepted_and_rejected_queries_do_not_consume_telegraph_stream() {
        let spec = StochasticSpec::Noise(TransientNoise {
            white_rms: 0.,
            step: 0.,
            alpha: 0.,
            flicker_drive_rms: 0.,
            rts_amplitude: 2.,
            capture_mean: 0.03,
            emission_mean: 0.05,
        });
        let a = StochasticWaveform::new(spec).unwrap();
        let expected: Vec<_> = (0..400)
            .map(|i| a.value_at(i as Real * 0.001, Limit::Right).unwrap())
            .collect();
        for (i, &value) in expected.iter().enumerate().rev() {
            assert_eq!(a.value_at(i as Real * 0.001, Limit::Right).unwrap(), value);
        }
        assert_eq!(a.value_at(0., Limit::Right).unwrap(), 0.);
        let mut iter = a.events(0., 0.2).unwrap();
        let mut count = 0;
        while let Some(event) = iter.next_event() {
            assert_eq!(event.kind, StochasticEventKind::Telegraph);
            count += 1;
        }
        assert!(count > 0);
    }
}
