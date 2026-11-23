//! Device-owned delay history and device-driven transient breakpoints.
//!
//! Some devices read their own accepted terminal waveforms a fixed time back:
//! the lossless transmission line (`T`, C `tra/`, `TRAdelays`) interpolates
//! the wave launched at the far port `TD` earlier, and the lossy line (`O`,
//! C `ltra/`) convolves its whole accepted history. That history is device
//! state in C (an instance array written by `TRAaccept`), but it is not a
//! fixed set of `CKTstate` slots: its length grows with the accepted points
//! and is pruned once older than the delay. The port therefore keeps it next
//! to the fixed slots, in [`crate::devices::StateHistory`], under the same
//! ownership rules (see `docs/port/TRANSIENT.md`, "Device-driven
//! breakpoints"):
//!
//! * a load (trial) only **reads** the history
//!   ([`crate::devices::DeviceState::delay_history`]);
//! * only an accepted transient point changes it, through a validated
//!   [`DelayUpdate`] that is applied after the fixed state is committed
//!   ([`crate::devices::Circuit::accept_transient_point`]). Rejected trials,
//!   Newton iterations and interpolated output samples never touch it;
//! * an accepted point may also request future breakpoints (C
//!   `CKTsetBreak` from `DEVaccept`), which the companion driver merges with
//!   the source breakpoints and the run's stop time;
//! * a converged trial may bound the next step (C `DEVtrunc`) through
//!   [`DelayLine::timestep_limit`].
//!
//! A device opts in through [`crate::devices::Device::delay_line`].

use std::collections::VecDeque;
use std::fmt;

use crate::maths::Vector;
use crate::primitives::{NodeId, Real, SpiceError, SpiceResult};

use crate::devices::traits::MnaUnknowns;

fn delay_error(message: impl Into<String>) -> SpiceError {
    SpiceError::Numerical {
        context: "delay history".to_owned(),
        message: message.into(),
    }
}

/// Accepted delay state, using the physical model's own time coordinates.
#[derive(Debug, Clone, PartialEq)]
pub enum DelayHistory {
    /// TRA/LTRA use continuous physical time and waveform samples.
    Waveform(WaveformHistory),
    /// TXL uses integer picosecond ticks and one accepted convolution state.
    Txl(Box<crate::devices::txl::TxlHistory>),
}

impl DelayHistory {
    /// Construct continuous-time waveform history.
    #[must_use]
    pub const fn waveform(width: usize) -> Self {
        Self::Waveform(WaveformHistory::new(width))
    }

    /// Access the continuous-time waveform history.
    ///
    /// # Errors
    /// This line uses a different history representation.
    pub fn as_waveform(&self) -> SpiceResult<&WaveformHistory> {
        match self {
            Self::Waveform(history) => Ok(history),
            Self::Txl(_) => Err(delay_error("expected continuous waveform history")),
        }
    }

    /// Access TXL history.
    ///
    /// # Errors
    /// This line uses a different history representation.
    pub fn as_txl(&self) -> SpiceResult<&crate::devices::txl::TxlHistory> {
        match self {
            Self::Txl(history) => Ok(history),
            Self::Waveform(_) => Err(delay_error("expected TXL tick history")),
        }
    }

    /// Number of accepted samples.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Waveform(history) => history.len(),
            Self::Txl(history) => history.len(),
        }
    }

    /// True before the line is initialized.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Check a typed update before any state is committed.
    ///
    /// # Errors
    /// Incompatible history or invalid update.
    pub fn check(&self, update: &DelayUpdate) -> SpiceResult<()> {
        match (self, update) {
            (Self::Waveform(history), DelayUpdate::Waveform(update)) => history.check(update),
            (Self::Txl(history), DelayUpdate::Txl(update)) => history.check(update),
            _ => Err(delay_error("incompatible delay update")),
        }
    }

    #[cfg(test)]
    pub(crate) fn apply(&mut self, update: &DelayUpdate) -> SpiceResult<()> {
        self.check(update)?;
        self.apply_validated(update);
        Ok(())
    }

    pub(super) fn apply_validated(&mut self, update: &DelayUpdate) {
        match (self, update) {
            (Self::Waveform(history), DelayUpdate::Waveform(update)) => {
                history.apply_validated(update);
            }
            (Self::Txl(history), DelayUpdate::Txl(update)) => history.apply_validated(update),
            _ => unreachable!("delay updates were validated before committing"),
        }
    }

    pub(crate) fn clear(&mut self) {
        match self {
            Self::Waveform(history) => history.clear(),
            Self::Txl(history) => history.clear(),
        }
    }
}

/// Model-specific update prepared before atomic transient acceptance.
#[derive(Debug, Clone, PartialEq)]
pub enum DelayUpdate {
    /// Continuous waveform update, possibly including future breakpoints.
    Waveform(WaveformUpdate),
    /// Integer-tick TXL update and its accepted convolution state.
    Txl(crate::devices::txl::TxlUpdate),
}

impl DelayUpdate {
    pub(super) const fn is_reset(&self) -> bool {
        match self {
            Self::Waveform(update) => update.reset.is_some(),
            Self::Txl(update) => update.is_reset(),
        }
    }

    pub(super) fn into_breakpoints(self) -> Vec<Real> {
        match self {
            Self::Waveform(update) => update.breakpoints,
            Self::Txl(_) => Vec::new(),
        }
    }
}

/// Accepted samples of one delay device, oldest first (C `TRAdelays`: rows of
/// `time, values...`). Times are finite and strictly increasing; every sample
/// holds [`Self::width`] finite values.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct WaveformHistory {
    width: usize,
    times: VecDeque<Real>,
    values: VecDeque<Real>,
}

impl WaveformHistory {
    /// An empty history with `width` values per sample.
    #[must_use]
    pub const fn new(width: usize) -> Self {
        Self {
            width,
            times: VecDeque::new(),
            values: VecDeque::new(),
        }
    }

    /// Values per sample.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Number of samples.
    #[must_use]
    pub fn len(&self) -> usize {
        self.times.len()
    }

    /// True before the transient start initialized the history.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    /// The time of sample `index` (0 is the oldest).
    #[must_use]
    pub fn time(&self, index: usize) -> Option<Real> {
        self.times.get(index).copied()
    }

    /// A width-checked, nonempty view of the accepted samples in logical
    /// ring order. The update transaction guarantees finite values and
    /// strictly increasing times.
    pub(super) fn samples<const N: usize>(&self) -> SpiceResult<DelaySamples<'_, N>> {
        if N == 0 || self.width != N || self.is_empty() {
            return Err(delay_error(format!(
                "expected nonempty history of width {N}, got width {} and length {}",
                self.width,
                self.len()
            )));
        }
        Ok(DelaySamples { history: self })
    }

    /// Value `component` of sample `index`.
    #[must_use]
    pub fn value(&self, index: usize, component: usize) -> Option<Real> {
        if index >= self.len() || component >= self.width {
            return None;
        }
        self.values.get(index * self.width + component).copied()
    }

    /// The newest sample's time.
    #[must_use]
    pub fn last_time(&self) -> Option<Real> {
        self.times.back().copied()
    }

    /// Checks that `update` can be applied: matching widths, finite values,
    /// strictly increasing times and a drop count that leaves the appended
    /// sample after every kept one.
    ///
    /// # Errors
    ///
    /// [`SpiceError::Numerical`] describing the first violation.
    pub fn check(&self, update: &WaveformUpdate) -> SpiceResult<()> {
        let check_sample = |time: Real, values: &[Real]| {
            if values.len() != self.width {
                return Err(delay_error(format!(
                    "sample has {} values, history has {}",
                    values.len(),
                    self.width
                )));
            }
            if !time.is_finite() || values.iter().any(|v| !v.is_finite()) {
                return Err(delay_error("nonfinite delay sample"));
            }
            Ok(())
        };
        let mut last = None;
        if let Some(reset) = &update.reset {
            for (time, values) in reset {
                check_sample(*time, values)?;
                if last.is_some_and(|previous| *time <= previous) {
                    return Err(delay_error("delay samples must have increasing times"));
                }
                last = Some(*time);
            }
        } else {
            if update.drop_front > self.len() {
                return Err(delay_error(format!(
                    "cannot drop {} of {} delay samples",
                    update.drop_front,
                    self.len()
                )));
            }
            last = self.last_time();
            if update.drop_front == self.len() {
                last = None;
            }
        }
        if let Some((time, values)) = &update.append {
            check_sample(*time, values)?;
            if last.is_some_and(|previous| *time <= previous) {
                return Err(delay_error(format!(
                    "appended delay sample at {time:e} is not after the last one"
                )));
            }
        }
        if update.breakpoints.iter().any(|t| !t.is_finite()) {
            return Err(delay_error("nonfinite breakpoint request"));
        }
        Ok(())
    }

    /// Checked standalone application for line-model unit tests. Production
    /// updates are committed together by StateHistory::accept_transaction.
    #[cfg(test)]
    pub(crate) fn apply(&mut self, update: &WaveformUpdate) -> SpiceResult<()> {
        self.check(update)?;
        self.apply_validated(update);
        Ok(())
    }

    /// Publish an update preflighted by the owning state transaction.
    /// No fallible validation remains after fixed state has been committed.
    pub(super) fn apply_validated(&mut self, update: &WaveformUpdate) {
        if let Some(reset) = &update.reset {
            self.times.clear();
            self.values.clear();
            for (time, values) in reset {
                self.times.push_back(*time);
                self.values.extend(values.iter().copied());
            }
        } else {
            self.times.drain(..update.drop_front);
            self.values.drain(..update.drop_front * self.width);
        }
        if let Some((time, values)) = &update.append {
            self.times.push_back(*time);
            self.values.extend(values.iter().copied());
        }
    }

    /// Forgets every sample (a restarted run).
    pub(crate) fn clear(&mut self) {
        self.times.clear();
        self.values.clear();
    }
}

/// Width-checked accepted history, without materializing a contiguous copy.
/// All indices are logical oldest-first indices, including after pruning.
#[derive(Clone, Copy)]
pub(super) struct DelaySamples<'a, const N: usize> {
    history: &'a WaveformHistory,
}

/// One complete accepted sample of a width-checked history.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct DelaySample<const N: usize> {
    pub(super) time: Real,
    pub(super) values: [Real; N],
}

impl<'a, const N: usize> DelaySamples<'a, N> {
    pub(super) const fn times(self) -> &'a VecDeque<Real> {
        &self.history.times
    }

    /// First logical index whose time is at or after the target. The
    /// returned index may equal the length. No monotonic query assumption.
    pub(super) fn lower_bound(self, target: Real) -> usize {
        self.history.times.partition_point(|&time| time < target)
    }

    /// First logical index strictly after the target, including exact knots
    /// in the prefix. Interpolation and pruning deliberately use different
    /// bounds. Both searches operate directly on the wrapped ring.
    pub(super) fn upper_bound(self, target: Real) -> usize {
        self.history.times.partition_point(|&time| time <= target)
    }

    /// Retrieve a fixed-size nonempty window ending at a logical index.
    /// Check bounds once, then copy only its samples, not the full history.
    pub(super) fn window<const K: usize>(self, end: usize) -> Option<[DelaySample<N>; K]> {
        if K == 0 || end >= self.history.len() {
            return None;
        }
        let start = (end + 1).checked_sub(K)?;
        Some(std::array::from_fn(|offset| {
            let index = start + offset;
            DelaySample {
                time: self.history.times[index],
                values: self.sample(index),
            }
        }))
    }

    /// Traverse complete sample rows directly from the two ring slices.
    /// A wrap can split one row; reconstruct only that row, without copying
    /// or indexing the rest of the history.
    pub(super) fn rows(self) -> impl DoubleEndedIterator<Item = [Real; N]> + 'a {
        let (first, second) = self.history.values.as_slices();
        let (first_rows, tail) = first.as_chunks::<N>();
        let (seam, second) = if tail.is_empty() {
            (None, second)
        } else {
            let (head, remaining) = second.split_at(N - tail.len());
            let seam = std::array::from_fn(|i| {
                if i < tail.len() {
                    tail[i]
                } else {
                    head[i - tail.len()]
                }
            });
            (Some(seam), remaining)
        };
        let (second_rows, remainder) = second.as_chunks::<N>();
        debug_assert_eq!(remainder, []);
        first_rows
            .iter()
            .copied()
            .chain(seam)
            .chain(second_rows.iter().copied())
    }

    pub(super) fn sample(self, index: usize) -> [Real; N] {
        let start = index * N;
        std::array::from_fn(|component| self.history.values[start + component])
    }
}

/// What one accepted transient point does to one device's [`WaveformHistory`].
///
/// Applied atomically after the fixed device state is committed: first
/// `reset` (the transient start) or `drop_front`, then `append`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct WaveformUpdate {
    /// Replace the whole history with these samples (C `MODEINITTRAN`
    /// initialization of `TRAdelays`).
    pub reset: Option<Vec<(Real, Vec<Real>)>>,
    /// Drop this many oldest samples (C `TRAaccept`'s shift).
    pub drop_front: usize,
    /// Append this sample.
    pub append: Option<(Real, Vec<Real>)>,
    /// Breakpoints requested for later times (C `CKTsetBreak`).
    pub breakpoints: Vec<Real>,
}

/// Transient step-size history supplied to delay devices.
///
/// The step sizes a delay device sees, C `CKTdeltaOld[0..3]`: during a trial
/// `[trial step, previous accepted step, the one before]`, at acceptance
/// `[accepted step, previous accepted step, the one before]`. Steps that do
/// not exist yet read as the run's maximum step, as `dctran.c` initializes
/// `CKTdeltaOld` with `CKTmaxStep`.
pub type DeltaOld = [Real; 3];

/// Everything a [`DelayLine`] sees at a transient start, an accepted point or
/// a converged trial.
#[derive(Debug, Clone)]
pub struct DelayContext<'a> {
    /// The accepted (or converged trial) solution.
    pub solution: &'a Vector,
    /// Node numbering, ground eliminated.
    pub unknowns: &'a MnaUnknowns,
    /// The accepted (or trial) time.
    pub time: Real,
    /// C `CKTdeltaOld[0..3]` ([`DeltaOld`]).
    pub steps: DeltaOld,
    /// C `CKTminBreak` as `TRAaccept` uses it: a sample closer than this to
    /// the previous one is not recorded. (The driver merges breakpoints with
    /// its own threshold.)
    pub min_break: Real,
    /// The run's relative tolerance (C `CKTreltol`), read by the lossy line's
    /// breakpoint test (`ltraacct.c`).
    pub reltol: Real,
    /// The run's absolute current tolerance (C `CKTabstol`).
    pub abstol: Real,
    /// The run starts from `uic` initial conditions (C `MODEUIC`).
    pub initial_conditions: bool,
    /// This device's branch-current rows, in order (C's `CKTstate`/RHS rows
    /// a delay line reads for its own currents).
    pub branches: std::ops::Range<usize>,
}

impl DelayContext<'_> {
    /// The solution's voltage at `node`, zero at ground.
    #[must_use]
    pub fn node_voltage(&self, node: NodeId) -> Real {
        self.unknowns
            .node_row(node)
            .and_then(|row| self.solution.get(row))
            .unwrap_or(0.)
    }
}

/// A device with an accepted-waveform delay history (see the module
/// documentation). Every method is a pure function of its arguments: the
/// device never mutates itself or the history.
pub trait DelayLine: fmt::Debug {
    /// Construct empty model-specific accepted state.
    fn new_history(&self) -> DelayHistory;

    /// The history at the transient start (`context.time` is 0), from the
    /// initial operating point or, under `uic`, the instance initial
    /// conditions. Must return a reset update.
    ///
    /// # Errors
    /// Device-specific failures.
    fn start(&self, context: &DelayContext<'_>) -> SpiceResult<DelayUpdate>;

    /// The update for an accepted point after the start.
    ///
    /// # Errors
    /// Device-specific failures (for instance a history too short).
    fn accept(
        &self,
        history: &DelayHistory,
        context: &DelayContext<'_>,
    ) -> SpiceResult<DelayUpdate>;

    /// An upper bound on the next step from a converged trial (C
    /// `DEVtrunc`), or `None`. It may be at or below zero, which rejects the
    /// trial like any truncation bound below `0.9 dt`.
    ///
    /// # Errors
    /// Device-specific failures.
    fn timestep_limit(
        &self,
        history: &DelayHistory,
        context: &DelayContext<'_>,
    ) -> SpiceResult<Option<Real>>;
}

#[cfg(test)]
mod tests {
    use super::{WaveformHistory, WaveformUpdate};
    use crate::primitives::Real;
    use std::collections::VecDeque;

    fn sample(time: f64, value: f64) -> (f64, Vec<f64>) {
        (time, vec![value, -value])
    }

    #[test]
    fn typed_samples_preserve_ring_order_after_pruning() {
        let mut history = WaveformHistory::new(2);
        for time in 0_i32..32 {
            history
                .apply(&WaveformUpdate {
                    append: Some(sample(f64::from(time), f64::from(time))),
                    drop_front: usize::from(time >= 4),
                    ..WaveformUpdate::default()
                })
                .unwrap();
            let expected = ((time - 3).max(0)..=time)
                .map(f64::from)
                .collect::<Vec<_>>();
            let view = history.samples::<2>().unwrap();
            assert_eq!(view.times().iter().copied().collect::<Vec<_>>(), expected);
            for (index, &time) in expected.iter().enumerate() {
                assert_eq!(view.sample(index), [time, -time]);
            }
            assert!(history.samples::<4>().is_err());
        }
        history.clear();
        assert!(history.samples::<2>().is_err());
    }

    #[test]
    fn slice_rows_preserve_a_split_sample_in_both_directions() {
        // A VecDeque can have a capacity that is not a multiple of the row
        // width. Construct a full logical history with a three-value seam.
        let mut values = VecDeque::with_capacity(35);
        let offset = values.capacity() - 3;
        values.resize(offset, -1.);
        values.drain(..offset - 1);
        values.extend((0..32).map(Real::from));
        assert_eq!(values.pop_front(), Some(-1.));
        assert_eq!(values.as_slices().0.len(), 3);
        let history = WaveformHistory {
            width: 4,
            times: (0..8).map(Real::from).collect(),
            values,
        };
        let view = history.samples::<4>().unwrap();
        let expected: Vec<_> = (0..8).map(|i| view.sample(i)).collect();
        assert_eq!(view.rows().collect::<Vec<_>>(), expected);
        assert_eq!(
            view.rows().rev().collect::<Vec<_>>(),
            expected.into_iter().rev().collect::<Vec<_>>()
        );
    }

    #[test]
    fn indexed_windows_preserve_strict_and_inclusive_boundaries_in_wrapped_ring() {
        let mut history = WaveformHistory::new(2);
        // Keep the most recent eight samples, forcing both deques to wrap.
        for tick in -8_i32..=4 {
            let time = f64::from(tick) / 4.;
            history
                .apply(&WaveformUpdate {
                    append: Some(sample(time, time)),
                    drop_front: usize::from(history.len() >= 8),
                    ..WaveformUpdate::default()
                })
                .unwrap();
        }
        assert!(!history.times.as_slices().1.is_empty());
        assert!(!history.values.as_slices().1.is_empty());
        let view = history.samples::<2>().unwrap();
        let times = view.times();
        let mut targets = vec![
            Real::NEG_INFINITY,
            -10.,
            -0.,
            0.,
            10.,
            Real::INFINITY,
            Real::NAN,
        ];
        for &t in times {
            targets.extend([t.next_down(), t, t.next_up()]);
        }
        // Repeated and decreasing queries model retries and rejected steps.
        targets.extend([times[6], times[2], times[6], times[0]]);
        let last = times.len() - 1;
        for target in targets {
            let old = |inclusive| {
                let mut index = 2;
                while index < last
                    && if inclusive {
                        times[index] <= target
                    } else {
                        times[index] < target
                    }
                {
                    index += 1;
                }
                index
            };
            let upper = view.upper_bound(target).clamp(2, last);
            let lower = view.lower_bound(target).clamp(2, last);
            assert_eq!(upper, old(true));
            assert_eq!(lower, old(false));
            let window = view.window::<3>(upper).unwrap();
            for (offset, sample) in window.into_iter().enumerate() {
                let index = upper - 2 + offset;
                assert_eq!(sample.time.to_bits(), times[index].to_bits());
                assert_eq!(
                    sample.values.map(Real::to_bits),
                    view.sample(index).map(Real::to_bits)
                );
            }
        }
        assert!(view.window::<0>(last).is_none());
        assert!(view.window::<3>(1).is_none());
        assert!(view.window::<1>(times.len()).is_none());
        assert!(view.window::<2>(usize::MAX).is_none());
    }

    /// A reproducible lookup-only microbenchmark; build an optimized test
    /// binary and run this test explicitly with --ignored --nocapture.
    #[test]
    #[ignore = "lookup microbenchmark, no timing assertion"]
    fn indexed_history_lookup_benchmark() {
        use std::{hint::black_box, time::Instant};
        for count in [1024_i32, 16_384, 262_144] {
            let mut history = WaveformHistory::new(2);
            history
                .apply(&WaveformUpdate {
                    reset: Some(
                        (0..count)
                            .map(|i| sample(Real::from(i), Real::from(i)))
                            .collect(),
                    ),
                    ..WaveformUpdate::default()
                })
                .unwrap();
            let view = history.samples::<2>().unwrap();
            let samples = view.times();
            let last = samples.len() - 1;
            let queries = (0_u32..2048)
                .map(|i| {
                    let tick = i.wrapping_mul(2_654_435_761) & (u32::try_from(count).unwrap() - 1);
                    Real::from(tick) + if i % 2 == 0 { 0. } else { 0.25 }
                })
                .collect::<Vec<_>>();
            let start = Instant::now();
            let mut expected = 0;
            for &query in &queries {
                let query = black_box(query);
                let mut upper = 2;
                while upper < last && samples[upper] <= query {
                    upper += 1;
                }
                let mut lower = 2;
                while lower < last && samples[lower] < query {
                    lower += 1;
                }
                expected += black_box(upper + lower);
            }
            let linear = start.elapsed();
            let start = Instant::now();
            let mut actual = 0;
            for &query in &queries {
                let query = black_box(query);
                actual += black_box(
                    view.upper_bound(query).clamp(2, last) + view.lower_bound(query).clamp(2, last),
                );
            }
            let indexed = start.elapsed();
            assert_eq!(actual, expected);
            println!(
                "{count} samples, {} queries: linear {linear:?}, indexed {indexed:?}, ratio {:.2}x",
                queries.len(),
                linear.as_secs_f64() / indexed.as_secs_f64()
            );
        }
    }

    #[test]
    fn updates_reset_drop_and_append_atomically() {
        let mut history = WaveformHistory::new(2);
        assert!(history.is_empty());
        history
            .apply(&WaveformUpdate {
                reset: Some(vec![sample(-2., 1.), sample(-1., 1.), sample(0., 1.)]),
                ..WaveformUpdate::default()
            })
            .unwrap();
        assert_eq!(history.len(), 3);
        assert_eq!(history.time(0), Some(-2.));
        assert_eq!(history.value(2, 1), Some(-1.));
        assert_eq!(history.value(2, 2), None);
        assert_eq!(history.value(usize::MAX, 0), None);
        assert_eq!(history.value(usize::MAX / 2 + 1, 0), None);
        history
            .apply(&WaveformUpdate {
                drop_front: 1,
                append: Some(sample(0.5, 3.)),
                ..WaveformUpdate::default()
            })
            .unwrap();
        assert_eq!(history.len(), 3);
        assert_eq!(history.time(0), Some(-1.));
        assert_eq!(history.last_time(), Some(0.5));
        assert_eq!(history.value(2, 0), Some(3.));
        let before = history.clone();
        for bad in [
            WaveformUpdate {
                append: Some(sample(0.5, 1.)),
                ..WaveformUpdate::default()
            },
            WaveformUpdate {
                drop_front: 4,
                ..WaveformUpdate::default()
            },
            WaveformUpdate {
                append: Some((1., vec![1.])),
                ..WaveformUpdate::default()
            },
            WaveformUpdate {
                append: Some(sample(1., f64::NAN)),
                ..WaveformUpdate::default()
            },
            WaveformUpdate {
                reset: Some(vec![sample(0., 1.), sample(0., 1.)]),
                ..WaveformUpdate::default()
            },
            WaveformUpdate {
                breakpoints: vec![f64::INFINITY],
                ..WaveformUpdate::default()
            },
        ] {
            assert!(history.apply(&bad).is_err(), "{bad:?}");
            assert_eq!(history, before);
        }
        history.clear();
        assert!(history.is_empty());
    }
}
