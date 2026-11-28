//! Trial versus accepted device state.
//!
//! ngspice keeps device history in rotating state vectors: `CKTstate0` is
//! written by every `<dev>load()` call during Newton iterations, and
//! `CKTstate1`, `CKTstate2`, … hold the values at previously accepted time
//! points. `dctran.c` rotates them only after a time point is accepted; a
//! rejected step simply reloads `CKTstate0`.
//!
//! The port makes that ownership explicit:
//!
//! - [`StateHistory`] owns the accepted vectors, most recent first. Only
//!   [`StateHistory::commit`] changes it, and it validates before mutating.
//! - [`TrialState`] is a disposable `CKTstate0`. Every load writes into a trial;
//!   dropping it is a rollback.
//! - [`DeviceState`] is the per-device window a [`crate::devices::Device`] sees while
//!   stamping: read-only accepted history, writable trial slots.
//!
//! Devices stamp through `&self`, so a trial cannot mutate a device either.
//! [`crate::devices::Circuit::accept_point`] runs every device's accept hook before the
//! commit, so a failing hook leaves the history untouched.
//!
//! # Newton phases
//!
//! C's `CKTstate0` also survives from one Newton iteration to the next, and
//! `NIiter` (`niiter.c`) tags each load with an initialization phase
//! (`MODEINITJCT`, `MODEINITFIX`, `MODEINITPRED`/`MODEINITTRAN`,
//! `MODEINITFLOAT`). Devices whose discrete state depends on the previous
//! iterate (the S/W switches, `swload.c`/`cswload.c`) need both. A trial
//! therefore carries its [`IterationPhase`], an optional read-only copy of the
//! previous load's trial values ([`DeviceState::iterate`]) and a
//! nonconvergence flag ([`DeviceState::report_nonconvergence`], C
//! `CKTnoncon++`) that the Newton driver reads. None of this is committed:
//! [`StateHistory::commit`] keeps only the values.
//!
//! # Junction limiting
//!
//! A Newton trial ([`StateHistory::trial_in`]) also allows *device limiting*
//! ([`DeviceState::device_limiting`]): diodes, BJTs and MOS1 then evaluate
//! their junction/FET voltages through C's `DEVpnjlim`/`DEVfetlim`/
//! `DEVlimvds` relative to the previous load's values, start a DC operating
//! point at C's `MODEINITJCT` voltages and report every limited load as
//! nonconvergent (see [`crate::devices::limiting`]). A plain [`StateHistory::trial`]
//! (any load outside a device-limited Newton solve) never limits: the device
//! is evaluated exactly at the supplied solution.
//!
//! # `uic` initial load
//!
//! [`TrialState::with_initial_conditions`] marks the single load C performs
//! for a `.tran ... uic` start (`MODETRANOP | MODEUIC | MODEINITJCT`, after
//! which `NIiter` returns without a solve): nonlinear devices then evaluate
//! at their instance initial-condition voltages (`DIOinitCond`,
//! `BJTicVBE`/`BJTicVCE`, `MOS1icVDS`/`VGS`/`VBS`), defaulted from the node
//! values of the supplied solution as `diogetic.c`/`bjtgetic.c`/`mos1ic.c`
//! do, independently of device limiting.
//!
//! # Delay histories
//!
//! Devices with a [`crate::devices::delay::DelayLine`] (transmission lines)
//! also own a variable-length [`DelayHistory`] of accepted samples, kept here
//! by device ordinal next to the fixed slots. Loads read it through
//! [`DeviceState::delay_history`]; only
//! [`crate::devices::Circuit::accept_transient_point`] changes it, after the
//! fixed state is committed, and [`StateHistory::commit`] alone never does.

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Arc;

use crate::devices::circuit::TopologyId;
use crate::devices::delay::{DelayHistory, DelayUpdate};

use crate::primitives::{Real, SpiceError, SpiceResult};

/// Number of accepted state vectors retained for transient integration.
///
/// Number of accepted state vectors retained (C `CKTstate1..CKTstate7` for
/// `maxord=6`; `cktsetup.c` allocates `max(2, maxord) + 2` vectors): enough
/// for Gear order-6 integration (six accepted charges) and its
/// truncation-error estimate (seven).
pub const ACCEPTED_DEPTH: usize = crate::maths::IntegrationMethod::MAX_GEAR_ORDER as usize + 1;

/// Why [`DeviceState::set`] refused `value` for `slot`: a nonfinite value,
/// otherwise an out-of-range slot.
#[cold]
#[inline(never)]
fn set_error(slot: usize, value: Real) -> SpiceError {
    if value.is_finite() {
        state_error(format!("state slot {slot} out of range"))
    } else {
        state_error(format!("nonfinite trial value for slot {slot}"))
    }
}

fn state_error(message: impl Into<String>) -> SpiceError {
    SpiceError::Numerical {
        context: "device state".to_owned(),
        message: message.into(),
    }
}

/// `MODEINITTRAN` writes into `CKTstate1` during a solve. A sparse, immutable
/// snapshot is shared by subsequent Newton trials until another write. Each
/// absolute slot keeps only its latest value.
#[derive(Debug, Clone, Default, PartialEq)]
struct State1Patches(Option<Arc<BTreeMap<usize, Real>>>);

impl State1Patches {
    fn get(&self, slot: usize) -> Option<Real> {
        self.0.as_deref()?.get(&slot).copied()
    }

    fn set(&mut self, slot: usize, value: Real) {
        Arc::make_mut(self.0.get_or_insert_with(Default::default)).insert(slot, value);
    }

    fn apply(&self, accepted: &mut [Arc<[Real]>]) {
        let Some(patches) = self.0.as_deref() else {
            return;
        };
        // C copies the first step's patched state1 into state2 and state3
        // before rotation. Do not detach older or unmodified vectors.
        for vector in accepted.iter_mut().take(3) {
            let values = Arc::make_mut(vector);
            for (&slot, &value) in patches {
                values[slot] = value;
            }
        }
    }
}

/// An immutable state vector tied to one finalized topology.
#[derive(Debug, Clone, PartialEq)]
pub struct StateSnapshot {
    pub(super) values: Arc<[Real]>,
    pub(super) topology: TopologyId,
}

impl StateSnapshot {
    /// The values in finalized state-row order.
    #[must_use = "the snapshot values are needed to inspect the stored device state"]
    pub fn values(&self) -> &[Real] {
        &self.values
    }
}

/// Publication mode for an accepted trial.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum HistoryPublication {
    Initialize,
    Rotate,
}

/// Shared finalized mapping from device ordinal to dense delay-history slot.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct DelayLayout(Arc<[Option<std::num::NonZeroUsize>]>);

impl DelayLayout {
    pub(super) fn new(devices: usize, ordinals: &[usize]) -> Self {
        if ordinals.is_empty() {
            return Self::default();
        }
        let mut slots = vec![None; devices];
        for (index, &ordinal) in ordinals.iter().enumerate() {
            slots[ordinal] = std::num::NonZeroUsize::new(index + 1);
        }
        Self(slots.into())
    }

    fn slot(&self, device: usize) -> Option<usize> {
        self.0
            .get(device)
            .copied()
            .flatten()
            .map(|slot| slot.get() - 1)
    }
}

/// One history per bound delay device, with reusable transaction scratch.
/// Scratch is always empty/unmarked outside a prepared transaction.
#[derive(Debug, Clone, Default, PartialEq)]
pub(super) struct DelayHistories {
    layout: DelayLayout,
    samples: Vec<DelayHistory>,
    seen: Vec<bool>,
    checked: Vec<usize>,
}

impl DelayHistories {
    pub(super) fn new(
        layout: DelayLayout,
        histories: impl IntoIterator<Item = DelayHistory>,
    ) -> Self {
        let samples: Vec<_> = histories.into_iter().collect();
        let count = samples.len();
        Self {
            layout,
            samples,
            seen: vec![false; count],
            checked: Vec::with_capacity(count),
        }
    }

    fn get(&self, device: usize) -> Option<&DelayHistory> {
        self.samples.get(self.layout.slot(device)?)
    }

    fn prepare(&mut self, updates: &[(usize, DelayUpdate)]) -> SpiceResult<()> {
        let result = updates.iter().try_for_each(|(device, update)| {
            let slot = self
                .layout
                .slot(*device)
                .ok_or_else(|| state_error(format!("device {device} has no delay history")))?;
            self.samples[slot].check(update)?;
            if self.seen[slot] {
                return Err(state_error(format!(
                    "duplicate delay update for device {device}"
                )));
            }
            self.seen[slot] = true;
            self.checked.push(slot);
            Ok(())
        });
        // Clear only marks actually visited, not all histories. This also
        // restores scratch after any failure before publication.
        for &slot in &self.checked {
            self.seen[slot] = false;
        }
        if result.is_err() {
            self.checked.clear();
        }
        result
    }

    fn discard(&mut self) {
        self.checked.clear();
    }

    fn apply(&mut self, updates: &[(usize, DelayUpdate)]) {
        for (slot, (_, update)) in self.checked.drain(..).zip(updates) {
            self.samples[slot].apply_validated(update);
        }
    }

    fn clear(&mut self) {
        for history in &mut self.samples {
            history.clear();
        }
    }
}

/// Accepted device state vectors, most recent first.
#[derive(Debug, Clone, PartialEq)]
pub struct StateHistory {
    len: usize,
    binding: Option<StateBinding>,
    revision: Arc<()>,
    accepted: Vec<Arc<[Real]>>,
    /// C's `CKTstate0` as the previous analysis of the deck left it (C runs
    /// every analysis on the same state vectors), read through
    /// [`DeviceState::carried`].
    carried: Option<Arc<[Real]>>,
    /// Delay histories by device ordinal (empty for circuits without delay
    /// lines).
    delays: DelayHistories,
}

/// Immutable positional interpretation shared by all histories of a circuit.
#[derive(Debug, Clone, PartialEq)]
struct StateBinding {
    topology: TopologyId,
    rows: Arc<[Range<usize>]>,
}

impl StateHistory {
    /// An empty history for `len` state slots.
    #[must_use = "the new history owns the accepted and trial state storage"]
    pub fn new(len: usize) -> Self {
        Self {
            len,
            binding: None,
            revision: Arc::new(()),
            accepted: Vec::new(),
            carried: None,
            delays: DelayHistories::default(),
        }
    }

    pub(super) const fn bound(
        topology: TopologyId,
        rows: Arc<[Range<usize>]>,
        len: usize,
        revision: Arc<()>,
        carried: Option<Arc<[Real]>>,
        delays: DelayHistories,
    ) -> Self {
        Self {
            len,
            binding: Some(StateBinding { topology, rows }),
            revision,
            accepted: Vec::new(),
            carried,
            delays,
        }
    }

    /// A local device-kernel history with previous-analysis values.
    #[cfg(test)]
    pub(super) fn with_carried(mut self, carried: Option<Vec<Real>>) -> Self {
        self.carried = carried
            .filter(|values| values.len() == self.len)
            .map(Into::into);
        self.revision = Arc::new(());
        self
    }

    pub(super) fn matches_topology(&self, topology: TopologyId) -> bool {
        self.binding
            .as_ref()
            .is_some_and(|binding| binding.topology == topology)
    }

    // The circuit verifies physical layout identity before this transfer.
    // DC never advances delay histories. Keep accepted vectors, but issue a
    // fresh revision so outstanding trials cannot cross a topology boundary.
    pub(super) fn inherit_dc_accepted(&mut self, previous: &Self) -> SpiceResult<()> {
        if self.len != previous.len
            || self.binding.as_ref().map(|b| &b.rows) != previous.binding.as_ref().map(|b| &b.rows)
        {
            return Err(state_error("incompatible DC state layout"));
        }
        self.accepted.clone_from(&previous.accepted);
        self.carried.clone_from(&previous.carried);
        self.revision = Arc::new(());
        Ok(())
    }

    /// Check identity without inspecting or requiring the trial's values.
    pub(super) fn check_trial(&self, trial: &TrialState) -> SpiceResult<()> {
        if trial.topology != self.binding.as_ref().map(|binding| binding.topology)
            || !Arc::ptr_eq(&self.revision, &trial.revision)
            || trial.values.len() != self.len
        {
            return Err(state_error(
                "trial belongs to a different state layout or accepted revision",
            ));
        }
        Ok(())
    }

    /// Retain a snapshot after the history is discarded, without copying its
    /// accepted vector. Subsequent history patches detach on mutation.
    pub(super) fn accepted_snapshot(&self, age: usize) -> Option<Arc<[Real]>> {
        age.checked_sub(1)
            .and_then(|index| self.accepted.get(index))
            .cloned()
    }

    /// The same state history with model-specific delay histories.
    #[cfg(test)]
    fn with_delay_lines(mut self, lines: impl IntoIterator<Item = (usize, DelayHistory)>) -> Self {
        let mut lines: Vec<_> = lines.into_iter().collect();
        lines.sort_unstable_by_key(|(device, _)| *device);
        let ordinals: Vec<_> = lines.iter().map(|(device, _)| *device).collect();
        let devices = ordinals.last().map_or(0, |ordinal| ordinal + 1);
        self.delays = DelayHistories::new(
            DelayLayout::new(devices, &ordinals),
            lines.into_iter().map(|(_, history)| history),
        );
        self.revision = Arc::new(());
        self
    }

    /// The delay history of the device with ordinal `device`, if it has one.
    #[must_use = "the delay history is needed to evaluate the specified device at earlier times"]
    pub fn delay(&self, device: usize) -> Option<&DelayHistory> {
        self.delays.get(device)
    }

    /// Checks one delay update per listed device without applying any.
    ///
    /// # Errors
    ///
    /// [`SpiceError::Numerical`] for a device without a delay history or an
    /// update [`DelayHistory::check`] rejects.
    pub fn check_delays(&mut self, updates: &[(usize, DelayUpdate)]) -> SpiceResult<()> {
        let result = self.delays.prepare(updates);
        self.delays.discard();
        result
    }

    /// Commit fixed state and delay histories as one transaction. The
    /// preparation callback cannot mutate history; the last fallible step
    /// is the circuit's accept hooks. Breakpoints retain the original
    /// device visitation order, independent of dense slot order.
    pub(super) fn accept_transaction(
        &mut self,
        trial: TrialState,
        publication: HistoryPublication,
        prepare: impl FnOnce(&Self, &TrialState) -> SpiceResult<Vec<(usize, DelayUpdate)>>,
        before_commit: impl FnOnce(&TrialState) -> SpiceResult<()>,
    ) -> SpiceResult<Vec<Real>> {
        self.check_publication(&trial, publication)?;
        let updates = prepare(self, &trial)?;
        self.delays.prepare(&updates)?;
        if let Err(error) = before_commit(&trial) {
            self.delays.discard();
            return Err(error);
        }
        self.commit_validated(trial, publication);
        self.delays.apply(&updates);
        Ok(updates
            .into_iter()
            .flat_map(|(_, update)| update.into_breakpoints())
            .collect())
    }

    /// Number of state slots per vector.
    #[must_use = "the state length determines the storage needed for device history"]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// True when the circuit has no state slots.
    #[must_use = "the result distinguishes zero-state circuits from those requiring history"]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// How many accepted vectors are available (at most [`ACCEPTED_DEPTH`]).
    #[must_use = "the accepted history depth bounds valid history ages"]
    pub const fn depth(&self) -> usize {
        self.accepted.len()
    }

    /// The accepted vector `age` points back: 1 is the latest (`CKTstate1`).
    #[must_use = "the returned snapshot is needed for time integration at the requested age"]
    pub fn accepted(&self, age: usize) -> Option<&[Real]> {
        age.checked_sub(1)
            .and_then(|index| self.accepted.get(index))
            .map(AsRef::as_ref)
    }

    /// A fresh trial in [`IterationPhase::Junction`] with no previous
    /// iterate and no device limiting: every device is evaluated exactly at
    /// the supplied solution. Its slots start as NaN, so a device that forgets
    /// to write one cannot have it committed.
    #[must_use = "the returned trial owns tentative state until the step is accepted"]
    pub fn trial(&self) -> TrialState {
        self.trial_reusing(None)
    }

    /// A fresh trial for a Newton load in `phase`, seeing `previous` (the
    /// trial of the preceding load of the same solve, C's `CKTstate0` before
    /// this load) read-only. Slots still start as NaN. Device limiting is
    /// allowed ([`TrialState::with_device_limiting`] turns it off for the
    /// port's legacy global-damping policy).
    ///
    /// # Errors
    ///
    /// [`SpiceError::Numerical`] when `previous` does not match the history.
    pub fn trial_in(
        &self,
        phase: IterationPhase,
        previous: Option<&TrialState>,
    ) -> SpiceResult<TrialState> {
        self.trial_in_reusing(phase, previous, None)
    }

    /// Reset a discarded Newton trial without borrowing or modifying any
    /// accepted/retained snapshot. Only an exclusively owned storage slice
    /// is recycled. Its old revision/topology is deliberately *not* trusted:
    /// all slots are overwritten with NaNs and the binding/revision rebound
    /// to this accepted history. The previous *active* trial still must
    /// match exactly; stale trials themselves can never be accepted.
    ///
    /// # Errors
    /// A previous trial from a foreign circuit or accepted revision.
    pub(crate) fn trial_in_reusing(
        &self,
        phase: IterationPhase,
        previous: Option<&TrialState>,
        spare: Option<TrialState>,
    ) -> SpiceResult<TrialState> {
        if let Some(previous) = previous {
            self.check_trial(previous)?;
        }
        Ok(self.recycle_trial(spare, LoadMode::newton(phase), previous))
    }

    /// Recycle an exact, independent evaluation trial without device limiting.
    pub(crate) fn trial_reusing(&self, spare: Option<TrialState>) -> TrialState {
        self.recycle_trial(spare, LoadMode::EXACT, None)
    }

    fn recycle_trial(
        &self,
        spare: Option<TrialState>,
        mode: LoadMode,
        previous: Option<&TrialState>,
    ) -> TrialState {
        let Some(mut reused) = spare.filter(|spare| spare.values.len() == self.len) else {
            return TrialState {
                topology: self.binding.as_ref().map(|binding| binding.topology),
                revision: Arc::clone(&self.revision),
                values: std::iter::repeat_n(Real::NAN, self.len).collect(),
                mode,
                previous: previous.map(|previous| Arc::clone(&previous.values)),
                nonconvergent: false,
                patches: previous.map_or_default(|previous| previous.patches.clone()),
            };
        };
        // The previous accepted revision invalidates *read access* to this
        // trial's old data, but not reuse of uniquely owned raw storage.
        // Rebind only after every slot has been overwritten. Arc::make_mut
        // would copy a retained snapshot unnecessarily merely to poison it.
        if let Some(values) = Arc::get_mut(&mut reused.values) {
            values.fill(Real::NAN);
        } else {
            reused.values = std::iter::repeat_n(Real::NAN, self.len).collect();
        }
        reused.topology = self.binding.as_ref().map(|binding| binding.topology);
        reused.revision = Arc::clone(&self.revision);
        reused.mode = mode;
        reused.previous = previous.map(|previous| Arc::clone(&previous.values));
        reused.nonconvergent = false;
        reused.patches = previous.map_or_default(|previous| previous.patches.clone());
        reused
    }

    /// Checks that `trial` could be committed.
    ///
    /// # Errors
    ///
    /// [`SpiceError::Numerical`] for a dimension mismatch or an unset or
    /// nonfinite slot.
    pub fn check(&self, trial: &TrialState) -> SpiceResult<()> {
        if trial.values.len() != self.len {
            return Err(state_error(format!(
                "trial has {} slots, history has {}",
                trial.values.len(),
                self.len
            )));
        }
        self.check_trial(trial)?;
        if let Some(slot) = trial.values.iter().position(|v| !v.is_finite()) {
            return Err(state_error(format!(
                "trial state slot {slot} is unset or nonfinite"
            )));
        }
        Ok(())
    }

    /// Commits `trial` as the newest accepted vector, dropping the oldest
    /// beyond [`ACCEPTED_DEPTH`]. Nothing changes on error.
    ///
    /// # Errors
    ///
    /// As [`Self::check`].
    pub fn commit(&mut self, trial: TrialState) -> SpiceResult<()> {
        self.check_publication(&trial, HistoryPublication::Rotate)?;
        self.commit_validated(trial, HistoryPublication::Rotate);
        Ok(())
    }

    fn check_publication(
        &self,
        trial: &TrialState,
        publication: HistoryPublication,
    ) -> SpiceResult<()> {
        self.check(trial)?;
        if publication == HistoryPublication::Initialize && !self.accepted.is_empty() {
            return Err(state_error(
                "initial publication requires empty accepted history",
            ));
        }
        Ok(())
    }

    fn commit_validated(&mut self, trial: TrialState, publication: HistoryPublication) {
        match publication {
            HistoryPublication::Initialize => self.accepted.resize(ACCEPTED_DEPTH, trial.values),
            HistoryPublication::Rotate => {
                trial.patches.apply(&mut self.accepted);
                self.accepted.insert(0, trial.values);
                self.accepted.truncate(ACCEPTED_DEPTH);
            }
        }
        self.revision = Arc::new(());
    }

    /// Publish the initial accepted point into all integration history slots.
    /// Each slot shares the same immutable state vector until modified.
    ///
    /// # Errors
    /// Invalid or stale trial, or a history that already contains accepted points.
    pub fn initialize(&mut self, trial: TrialState) -> SpiceResult<()> {
        self.check_publication(&trial, HistoryPublication::Initialize)?;
        self.commit_validated(trial, HistoryPublication::Initialize);
        Ok(())
    }

    /// Re-anchor integration history at an accepted right-limit event.
    /// No new physical point or device hook is published. Only the
    /// pre-event integration stencil is discarded.
    pub(crate) fn restart_integration(&mut self) -> SpiceResult<()> {
        let latest = self
            .accepted
            .first()
            .cloned()
            .ok_or_else(|| state_error("cannot restart empty integration history"))?;
        self.accepted.clear();
        self.accepted.resize(ACCEPTED_DEPTH, latest);
        self.revision = Arc::new(());
        Ok(())
    }

    /// Begin a new inner DC sweep from the state just accepted.
    pub(crate) fn retain_latest(&mut self) {
        self.accepted.truncate(1);
        self.revision = Arc::new(());
        self.delays.clear();
    }

    /// Fill a verified absolute slot of a candidate point before acceptance.
    pub(crate) fn set_candidate_slot(
        &self,
        trial: &mut TrialState,
        slot: usize,
        value: Real,
    ) -> SpiceResult<()> {
        self.check_trial(trial)?;
        if !value.is_finite() {
            return Err(state_error("invalid candidate-state override"));
        }
        let target = Arc::make_mut(&mut trial.values)
            .get_mut(slot)
            .ok_or_else(|| set_error(slot, value))?;
        *target = value;
        Ok(())
    }

    /// Edit only a verified device-local candidate window. Accepted history
    /// remains immutable; a failed event probe discards this candidate.
    pub(crate) fn edit_candidate<T>(
        &self,
        trial: &mut TrialState,
        range: Range<usize>,
        edit: impl FnOnce(&mut [Real]) -> SpiceResult<T>,
    ) -> SpiceResult<T> {
        self.check_trial(trial)?;
        let slots = Arc::make_mut(&mut trial.values)
            .get_mut(range)
            .ok_or_else(|| state_error("invalid candidate event state range"))?;
        let result = edit(slots)?;
        if slots.iter().any(|value| !value.is_finite()) {
            return Err(state_error("nonfinite candidate event state"));
        }
        Ok(result)
    }

    /// Forgets every accepted vector and delay sample, e.g. when a run
    /// restarts.
    pub fn clear(&mut self) {
        self.accepted.clear();
        self.revision = Arc::new(());
        self.delays.clear();
    }

    /// The per-device window for slots `range` of `trial`.
    ///
    /// # Errors
    ///
    /// [`SpiceError::Numerical`] when `trial` or `range` do not fit.
    pub fn device<'a>(
        &'a self,
        trial: &'a mut TrialState,
        range: Range<usize>,
    ) -> SpiceResult<DeviceState<'a>> {
        if self.binding.is_some() {
            return Err(state_error(
                "circuit-bound device state requires a device ordinal",
            ));
        }
        self.window(trial, range, None)
    }

    /// Get a circuit-bound device window from its finalized ordinal.
    ///
    /// The slot range and delay ordinal cannot be supplied independently.
    ///
    /// # Errors
    /// Unbound history, unknown device, or foreign/stale trial.
    pub fn device_of<'a>(
        &'a self,
        trial: &'a mut TrialState,
        device: usize,
    ) -> SpiceResult<DeviceState<'a>> {
        let range = self
            .binding
            .as_ref()
            .and_then(|binding| binding.rows.get(device))
            .cloned()
            .ok_or_else(|| state_error("device ordinal not in finalized state layout"))?;
        self.window(trial, range, self.delays.get(device))
    }

    /// Disposable local window with float-mode trial semantics and limiting
    /// disabled. All source state is read-only; only one device's trial
    /// slots are allocated. Writes cannot reach the accepted history.
    pub(super) fn isolated_device<'a>(
        &'a self,
        scratch: &'a mut IsolatedState,
        previous: Option<&'a TrialState>,
        device: usize,
    ) -> SpiceResult<DeviceState<'a>> {
        if let Some(previous) = previous {
            self.check_trial(previous)?;
        }
        let range = self
            .binding
            .as_ref()
            .and_then(|binding| binding.rows.get(device))
            .cloned()
            .ok_or_else(|| state_error("device ordinal not in finalized state layout"))?;
        scratch.values.resize(range.len(), Real::NAN);
        scratch.values.fill(Real::NAN);
        scratch.nonconvergent = false;
        scratch.patches = previous.map_or_default(|trial| trial.patches.clone());
        let mut mode = LoadMode::newton(IterationPhase::Float);
        mode.set(LoadFlag::DeviceLimiting, false);
        let mut accepted = [&[][..]; ACCEPTED_DEPTH];
        for (slot, vector) in accepted.iter_mut().zip(&self.accepted) {
            *slot = &vector[range.clone()];
        }
        Ok(DeviceState {
            trial: &mut scratch.values,
            accepted,
            depth: self.accepted.len(),
            mode,
            previous: previous.map(|trial| &trial.values[range.clone()]),
            carried: self.carried.as_deref().map(|vector| &vector[range.clone()]),
            nonconvergent: Some(&mut scratch.nonconvergent),
            offset: range.start,
            patches: Some(&mut scratch.patches),
            delay: self.delays.get(device),
            ltra: None,
        })
    }

    fn window<'a>(
        &'a self,
        trial: &'a mut TrialState,
        range: Range<usize>,
        delay: Option<&'a DelayHistory>,
    ) -> SpiceResult<DeviceState<'a>> {
        self.check_trial(trial)?;
        let complete_trial = trial.values.len() == self.len;
        let ordered_range = range.start <= range.end;
        let bounded_range = range.end <= self.len;
        if !complete_trial || !ordered_range || !bounded_range {
            return Err(state_error("device state range out of bounds"));
        }
        let mut accepted: [&[Real]; ACCEPTED_DEPTH] = [&[]; ACCEPTED_DEPTH];
        for (slot, vector) in accepted.iter_mut().zip(&self.accepted) {
            *slot = &vector[range.clone()];
        }
        let TrialState {
            values,
            mode,
            previous,
            nonconvergent,
            patches,
            ..
        } = trial;
        let carried = self.carried.as_deref().map(|vector| &vector[range.clone()]);
        let previous = previous.as_deref().map(|vector| &vector[range.clone()]);
        Ok(DeviceState {
            previous,
            carried,
            trial: &mut Arc::make_mut(values)[range.clone()],
            accepted,
            depth: self.accepted.len(),
            mode: *mode,
            nonconvergent: Some(nonconvergent),
            offset: range.start,
            patches: Some(patches),
            delay,
            ltra: None,
        })
    }
}

/// The Newton initialization phase of a load (C `MODEINITF` bits set by
/// `NIiter` in `niiter.c`, `dctran.c` and `dctrcurv.c`).
///
/// Only devices with discrete, iterate-dependent state read it; everything
/// else stamps identically in every phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IterationPhase {
    /// First load of a DC operating-point solve (`MODEINITJCT`); also the
    /// phase of loads outside Newton iteration.
    #[default]
    Junction,
    /// Later DC operating-point loads until the iterate first converges
    /// (`MODEINITFIX`).
    Fix,
    /// First load of a transient timepoint or of a warm-started DC sweep
    /// point (`MODEINITTRAN`/`MODEINITPRED`), where the accepted history holds
    /// the previous point.
    Predict,
    /// Every other load (`MODEINITFLOAT`).
    Float,
}

/// The independent C load-mode switches. Keep convergence reporting
/// separate: it is mutable output of a load, not an input mode.
#[derive(Debug, Clone, Copy)]
#[repr(u8)]
enum LoadFlag {
    DeviceLimiting = 1 << 0,
    InitialConditions = 1 << 1,
    ReferenceJacobian = 1 << 2,
    InitialTransient = 1 << 3,
    Uic = 1 << 4,
    PreparedExact = 1 << 5,
}

/// Immutable metadata copied from the Newton trial into each device window.
/// A new Newton load starts with fresh options; only iterate values and
/// staged state1 patches are inherited from the preceding trial.
#[derive(Debug, Clone, Copy, PartialEq)]
struct LoadMode {
    phase: IterationPhase,
    flags: u8,
    tolerances: Option<(Real, Real)>,
}

impl LoadMode {
    const EXACT: Self = Self {
        phase: IterationPhase::Junction,
        flags: 0,
        tolerances: None,
    };

    const fn newton(phase: IterationPhase) -> Self {
        // NIiter factors C's DEVload matrix.
        Self {
            phase,
            flags: LoadFlag::DeviceLimiting as u8 | LoadFlag::ReferenceJacobian as u8,
            tolerances: None,
        }
    }

    const fn enabled(self, flag: LoadFlag) -> bool {
        self.flags & flag as u8 != 0
    }

    const fn set(&mut self, flag: LoadFlag, enabled: bool) {
        if enabled {
            self.flags |= flag as u8;
        } else {
            self.flags &= !(flag as u8);
        }
    }
}

/// The trial state vector (C `CKTstate0`) for one load.
///
/// Newton iterations share read-only snapshots of the preceding load instead of copying all
/// its state slots; mutating a cloned trial detaches its own vector.
#[derive(Debug, Clone, PartialEq)]
pub struct TrialState {
    topology: Option<TopologyId>,
    revision: Arc<()>,
    values: Arc<[Real]>,
    mode: LoadMode,
    previous: Option<Arc<[Real]>>,
    nonconvergent: bool,
    /// `MODEINITTRAN` writes into `CKTstate1` (absolute slot, value), seen by
    /// the later loads of the same solve and applied when the trial is
    /// committed ([`DeviceState::patch_state1`]).
    patches: State1Patches,
}

impl TrialState {
    /// Once this trial's load and device convergence check have completed,
    /// later Newton iterations need only its values and staged state1 patches.
    /// Release its predecessor snapshot to make that older, discarded state
    /// allocation uniquely reusable. Accepted/returned trials retain their
    /// complete history because this is only called on continuing passes.
    pub(crate) fn release_predecessor(&mut self) {
        self.previous = None;
    }

    /// Retain this load's values without copying its state slots.
    ///
    /// # Errors
    /// The trial is unbound or contains unset or nonfinite values.
    pub fn snapshot(&self) -> SpiceResult<StateSnapshot> {
        let topology = self
            .topology
            .ok_or_else(|| state_error("unbound trial has no circuit snapshot"))?;
        if self.values.iter().any(|value| !value.is_finite()) {
            return Err(state_error(
                "trial snapshot contains unset or nonfinite values",
            ));
        }
        Ok(StateSnapshot {
            topology,
            values: Arc::clone(&self.values),
        })
    }

    /// The same trial carrying the Newton solve's `reltol` and current
    /// `abstol`, which C's device convergence tests (`DEVconvTest`, run by
    /// `NIconvTest`) use; see [`DeviceState::convergence_tolerances`].
    #[must_use = "discarding the returned trial loses the requested convergence tolerances"]
    pub const fn with_convergence_tolerances(mut self, reltol: Real, abstol: Real) -> Self {
        self.mode.tolerances = Some((reltol, abstol));
        self
    }

    /// The same trial marked as part of a `uic` transient (C `MODEUIC`),
    /// see [`DeviceState::uic`].
    #[must_use = "discarding the returned trial loses the UIC load setting"]
    pub const fn with_uic(mut self, enabled: bool) -> Self {
        self.mode.set(LoadFlag::Uic, enabled);
        self
    }

    /// The same trial marked as the first transient timepoint's loads
    /// (`MODEINITTRAN` rather than `MODEINITPRED` in
    /// [`IterationPhase::Predict`]); see [`DeviceState::initial_transient`].
    #[must_use = "discarding the returned trial loses the initial transient load setting"]
    pub const fn with_initial_transient(mut self, enabled: bool) -> Self {
        self.mode.set(LoadFlag::InitialTransient, enabled);
        self
    }

    /// Keep prepared-kernel evaluation exact while explicit bias constraints apply.
    pub(crate) const fn with_prepared_exact(mut self, enabled: bool) -> Self {
        self.mode.set(LoadFlag::PreparedExact, enabled);
        self
    }

    /// The same trial marked (or unmarked) as C's `uic` initial load; see
    /// the module documentation and [`DeviceState::initial_conditions`].
    #[must_use = "discarding the returned trial loses the initial conditions setting"]
    pub const fn with_initial_conditions(mut self, enabled: bool) -> Self {
        self.mode.set(LoadFlag::InitialConditions, enabled);
        self
    }

    /// Whether this is the `uic` initial load.
    #[must_use = "the flag determines whether initial-condition load semantics apply"]
    pub const fn initial_conditions(&self) -> bool {
        self.mode.enabled(LoadFlag::InitialConditions)
    }

    /// The same trial asking devices to stamp C's own `DEVload` matrix where
    /// the port's Newton Jacobian deliberately differs from it (see
    /// [`DeviceState::c_jacobian`]). Newton solves leave it off; `.tf`
    /// (`tfanal.c`), which solves with the matrix C's last `CKTop` load left,
    /// turns it on.
    #[must_use = "discarding the returned trial loses the reference Jacobian selection"]
    pub const fn with_c_jacobian(mut self, enabled: bool) -> Self {
        self.mode.set(LoadFlag::ReferenceJacobian, enabled);
        self
    }

    /// The same trial with device limiting allowed or forbidden (see
    /// [`DeviceState::device_limiting`]). Newton drivers using the legacy
    /// global voltage-step damping forbid it so devices load exactly.
    #[must_use = "discarding the returned trial loses the device limiting selection"]
    pub const fn with_device_limiting(mut self, enabled: bool) -> Self {
        self.mode.set(LoadFlag::DeviceLimiting, enabled);
        self
    }

    /// Whether devices may limit their junction voltages in this load.
    #[must_use = "the flag determines whether nonlinear device limiting applies"]
    pub const fn device_limiting(&self) -> bool {
        self.mode.enabled(LoadFlag::DeviceLimiting)
    }

    /// The Newton phase this trial was loaded in.
    #[must_use = "the iteration phase determines which trial load rules apply"]
    pub const fn phase(&self) -> IterationPhase {
        self.mode.phase
    }

    /// True when a device reported that this load changed a discrete state
    /// relative to the previous iterate (C `CKTnoncon++`), so the solve must
    /// not be declared converged on it.
    #[must_use = "the nonconvergence flag determines whether the trial may be accepted"]
    pub const fn is_nonconvergent(&self) -> bool {
        self.nonconvergent
    }

    /// The trial values; unset slots are NaN.
    #[must_use = "the trial values are needed to inspect tentative device history"]
    pub fn values(&self) -> &[Real] {
        &self.values
    }

    /// The slots of one device, e.g. for an accept hook.
    #[must_use = "the returned slice identifies the requested valid trial-state window"]
    pub fn slice(&self, range: Range<usize>) -> Option<&[Real]> {
        self.values.get(range)
    }
}

/// Reusable, device-local disposable state storage for isolated asks.
#[derive(Debug, Default)]
pub(super) struct IsolatedState {
    values: Vec<Real>,
    nonconvergent: bool,
    patches: State1Patches,
}

impl IsolatedState {
    pub(super) fn clear(&mut self) {
        self.values.clear();
        self.patches = State1Patches::default();
        self.nonconvergent = false;
    }
}

/// One device's view of trial and accepted state while it stamps.
#[derive(Debug)]
pub struct DeviceState<'a> {
    trial: &'a mut [Real],
    accepted: [&'a [Real]; ACCEPTED_DEPTH],
    depth: usize,
    mode: LoadMode,
    previous: Option<&'a [Real]>,
    carried: Option<&'a [Real]>,
    nonconvergent: Option<&'a mut bool>,
    /// This window's first absolute slot.
    offset: usize,
    patches: Option<&'a mut State1Patches>,
    delay: Option<&'a DelayHistory>,
    ltra: Option<(&'a mut super::ltra::LtraWorkspace, &'a StateHistory, usize)>,
}

impl<'a> DeviceState<'a> {
    /// A window with no slots, for loads that track no state.
    #[must_use = "the returned empty state is needed for loads with no history slots"]
    pub const fn none() -> DeviceState<'static> {
        DeviceState {
            trial: &mut [],
            accepted: [&[]; ACCEPTED_DEPTH],
            depth: 0,
            mode: LoadMode::EXACT,
            previous: None,
            carried: None,
            nonconvergent: None,
            offset: 0,
            patches: None,
            delay: None,
            ltra: None,
        }
    }

    /// C's `MODEINITTRAN` write `*(CKTstate1 + slot) = value`: later loads of
    /// this solve read it through [`Self::accepted`] with age 1, and it
    /// becomes part of the history when the point is accepted.
    ///
    /// # Errors
    /// An out-of-range slot or a nonfinite value.
    pub fn patch_state1(&mut self, slot: usize, value: Real) -> SpiceResult<()> {
        if slot >= self.trial.len() || !value.is_finite() {
            return Err(state_error("invalid CKTstate1 write"));
        }
        if let Some(patches) = self.patches.as_deref_mut() {
            patches.set(self.offset + slot, value);
        }
        Ok(())
    }

    /// The device's accepted delay history (read-only), when it has one and
    /// the window was opened with [`StateHistory::device_of`].
    #[must_use = "the returned history supplies delayed samples for this device"]
    pub const fn delay_history(&self) -> Option<&'a DelayHistory> {
        self.delay
    }

    pub(super) fn with_ltra(
        mut self,
        workspace: Option<&'a mut super::ltra::LtraWorkspace>,
        accepted: &'a StateHistory,
        ordinal: usize,
    ) -> Self {
        self.ltra = workspace.map(|workspace| (workspace, accepted, ordinal));
        self
    }

    pub(super) fn ltra_workspace(
        &mut self,
    ) -> Option<(&mut super::ltra::LtraWorkspace, &'a StateHistory, usize)> {
        self.ltra
            .as_mut()
            .map(|(workspace, accepted, ordinal)| (&mut **workspace, *accepted, *ordinal))
    }

    /// The `(reltol, abstol)` of the Newton solve this load belongs to, when
    /// the driver supplied them
    /// ([`TrialState::with_convergence_tolerances`]).
    #[must_use = "the returned tolerances determine the device convergence threshold"]
    pub const fn convergence_tolerances(&self) -> Option<(Real, Real)> {
        self.mode.tolerances
    }

    /// The Newton phase of this load.
    #[must_use = "the iteration phase determines which local load rules apply"]
    pub const fn phase(&self) -> IterationPhase {
        self.mode.phase
    }

    /// Whether this load belongs to a device-limited Newton solve: a
    /// nonlinear device may then evaluate C's limited junction voltages
    /// instead of the supplied solution (and must report the load as
    /// nonconvergent when it does). `false` for exact loads.
    #[must_use = "the flag determines whether the device should apply nonlinear limiting"]
    pub const fn device_limiting(&self) -> bool {
        self.mode.enabled(LoadFlag::DeviceLimiting)
    }

    /// Whether a [`IterationPhase::Predict`] load is `MODEINITTRAN` (the first
    /// transient timepoint, junction voltages from `CKTstate1` unlimited)
    /// rather than `MODEINITPRED` (`DEVpred` extrapolation, limited).
    #[must_use = "the flag determines whether the first transient step is being loaded"]
    pub const fn initial_transient(&self) -> bool {
        self.mode.enabled(LoadFlag::InitialTransient)
    }

    /// Whether the transient runs with `uic` (C `MODEUIC`): in the first
    /// step's `MODEINITTRAN` load capacitors and inductors start from their
    /// `ic=` values.
    #[must_use = "the flag determines whether UIC initialization applies"]
    pub const fn uic(&self) -> bool {
        self.mode.enabled(LoadFlag::Uic)
    }

    pub(crate) const fn prepared_exact(&self) -> bool {
        self.mode.enabled(LoadFlag::PreparedExact)
    }

    /// C `DEVpred` (`devsup.c`): `(1 + r) state1 - r state2` with
    /// `r = CKTdelta / CKTdeltaOld[1]`; `state2` is `state1` when only one
    /// accepted vector exists (dctran.c copies state1 into state2 after the
    /// first step).
    #[must_use = "the predicted value provides the extrapolated history for the trial step"]
    #[inline]
    pub fn predicted(&self, slot: usize, ratio: Real) -> Option<Real> {
        let s1 = self.accepted(1, slot)?;
        let s2 = self.accepted(2, slot).unwrap_or(s1);
        let latest = (1. + ratio) * s1;
        let preceding = ratio * s2;
        Some(latest - preceding)
    }

    /// Whether this is C's `uic` initial load
    /// ([`TrialState::with_initial_conditions`]): nonlinear devices evaluate
    /// at their instance initial-condition voltages.
    #[must_use = "the flag selects initial-condition behavior for the device load"]
    pub const fn initial_conditions(&self) -> bool {
        self.mode.enabled(LoadFlag::InitialConditions)
    }

    /// Whether the load must stamp C's `DEVload` matrix rather than the
    /// port's exact Newton Jacobian where the two deliberately differ
    /// ([`TrialState::with_c_jacobian`]). The equations, residual and fixed
    /// point are unchanged; only the matrix differs. Today only the BJT reads
    /// it (`bjtload.c` stamps the bias-dependent base resistance as the
    /// conductance `gx` alone).
    #[must_use = "the flag selects the reference Jacobian stamping behavior"]
    pub const fn c_jacobian(&self) -> bool {
        self.mode.enabled(LoadFlag::ReferenceJacobian)
    }

    /// The value of `slot` written by the previous load of the same Newton
    /// solve (C `CKTstate0` before this load), or `None` for the first load.
    #[must_use = "the previous Newton iterate is needed for local convergence checks"]
    pub fn iterate(&self, slot: usize) -> Option<Real> {
        self.previous
            .and_then(|previous| previous.get(slot))
            .copied()
    }

    /// The value of `slot` in the state vector an earlier analysis of the
    /// deck left behind (C's `CKTstate0` at the start of this analysis, which
    /// C does not reset between analyses), if any.
    #[must_use = "the carried value supplies the device state from the preceding load"]
    pub fn carried(&self, slot: usize) -> Option<Real> {
        self.carried.and_then(|carried| carried.get(slot)).copied()
    }

    /// Marks this load as not converged (C `CKTnoncon++`): a discrete state
    /// changed since the previous iterate, so one more iteration is needed.
    ///
    /// # Errors
    ///
    /// [`SpiceError::Numerical`] for a window that tracks no trial.
    pub fn report_nonconvergence(&mut self) -> SpiceResult<()> {
        let flag = self
            .nonconvergent
            .as_deref_mut()
            .ok_or_else(|| state_error("nonconvergence reported outside a trial"))?;
        *flag = true;
        Ok(())
    }

    /// Number of slots this device owns.
    #[must_use = "the local state length bounds valid device slots"]
    pub const fn len(&self) -> usize {
        self.trial.len()
    }

    /// True when the device owns no slots.
    #[must_use = "the result distinguishes a stateless device from one with history"]
    pub const fn is_empty(&self) -> bool {
        self.trial.is_empty()
    }

    /// How many accepted points are available (0 before the first accepted
    /// point, or when the load tracks no history).
    #[must_use = "the local history depth bounds accepted state ages"]
    pub const fn depth(&self) -> usize {
        self.depth
    }

    /// The trial value of `slot`, NaN when not yet written in this load.
    #[must_use = "the returned value provides the tentative state for this slot"]
    pub fn trial(&self, slot: usize) -> Option<Real> {
        self.trial.get(slot).copied()
    }

    /// Writes the trial value of `slot`.
    ///
    /// # Errors
    ///
    /// [`SpiceError::Numerical`] for an out-of-range slot or a nonfinite value.
    #[inline]
    pub fn set(&mut self, slot: usize, value: Real) -> SpiceResult<()> {
        match self.trial.get_mut(slot) {
            Some(entry) if value.is_finite() => {
                *entry = value;
                Ok(())
            }
            _ => Err(set_error(slot, value)),
        }
    }

    /// The accepted value of `slot`, `age` points back (1 is the latest).
    #[must_use = "the returned value provides the accepted state for this local slot and age"]
    #[inline]
    pub fn accepted(&self, age: usize, slot: usize) -> Option<Real> {
        // Validate the local index before translating it into an absolute
        // slot: a patch in another device's window is never visible here.
        if slot >= self.trial.len() {
            return None;
        }
        if age == 1
            && let Some(patches) = self.patches.as_deref()
            && let Some(value) = patches.get(self.offset + slot)
        {
            return Some(value);
        }
        age.checked_sub(1)
            .and_then(|index| self.accepted.get(index))
            .and_then(|vector| vector.get(slot))
            .copied()
    }
}

#[cfg(test)]
mod tests {
    use super::{ACCEPTED_DEPTH, DeviceState, IterationPhase, StateHistory};
    use std::sync::Arc;

    #[test]
    fn load_mode_defaults_and_overrides_are_shared_by_every_window() {
        let history = StateHistory::new(2);
        let mut exact = history.trial();
        let window = history.device(&mut exact, 0..1).unwrap();
        assert_eq!(window.phase(), IterationPhase::Junction);
        assert!(!window.device_limiting());
        assert!(!window.c_jacobian());
        assert!(!window.initial_transient());
        assert!(!window.initial_conditions());
        assert!(!window.uic());
        assert_eq!(window.convergence_tolerances(), None);

        let mut trial = history
            .trial_in(IterationPhase::Predict, Some(&exact))
            .unwrap()
            .with_device_limiting(false)
            .with_c_jacobian(false)
            .with_initial_transient(true)
            .with_initial_conditions(true)
            .with_uic(true)
            .with_convergence_tolerances(1e-3, 1e-12);
        assert_eq!(trial.phase(), IterationPhase::Predict);
        assert!(!trial.device_limiting());
        assert!(trial.initial_conditions());
        for range in [0..1, 1..2] {
            let window = history.device(&mut trial, range).unwrap();
            assert_eq!(window.phase(), IterationPhase::Predict);
            assert!(!window.device_limiting());
            assert!(!window.c_jacobian());
            assert!(window.initial_transient());
            assert!(window.initial_conditions());
            assert!(window.uic());
            assert_eq!(window.convergence_tolerances(), Some((1e-3, 1e-12)));
        }
        let mut next = history
            .trial_in(IterationPhase::Float, Some(&trial))
            .unwrap();
        let window = history.device(&mut next, 0..1).unwrap();
        assert_eq!(window.phase(), IterationPhase::Float);
        assert!(window.device_limiting());
        assert!(window.c_jacobian());
        assert!(!window.initial_transient());
        assert!(!window.initial_conditions());
        assert!(!window.uic());
        assert_eq!(window.convergence_tolerances(), None);
    }

    #[test]
    fn commits_rotate_and_are_bounded() {
        let mut history = StateHistory::new(2);
        assert_eq!(history.depth(), 0);
        assert_eq!(history.accepted(1), None);
        for step in 0_u32..9 {
            let mut trial = history.trial();
            let mut device = history.device(&mut trial, 0..2).unwrap();
            assert_eq!(device.depth(), (step as usize).min(ACCEPTED_DEPTH));
            if step > 0 {
                assert_eq!(device.accepted(1, 0), Some(f64::from(step - 1)));
            }
            device.set(0, f64::from(step)).unwrap();
            device.set(1, -f64::from(step)).unwrap();
            history.commit(trial).unwrap();
        }
        assert_eq!(
            ACCEPTED_DEPTH, 7,
            "Gear order 6 needs seven accepted vectors"
        );
        assert_eq!(history.depth(), ACCEPTED_DEPTH);
        assert_eq!(history.accepted(1), Some(&[8.0, -8.0][..]));
        assert_eq!(history.accepted(3), Some(&[6.0, -6.0][..]));
        assert_eq!(history.accepted(7), Some(&[2.0, -2.0][..]));
        assert_eq!(history.accepted(0), None);
        assert_eq!(history.accepted(8), None);
        history.clear();
        assert_eq!(history.depth(), 0);
    }

    #[test]
    fn disposable_trial_reuse_poison_resets_every_unwritten_slot() {
        let history = StateHistory::new(3);
        let mut initial = history.trial_in(IterationPhase::Junction, None).unwrap();
        for slot in 0..3 {
            history
                .device(&mut initial, slot..slot + 1)
                .unwrap()
                .set(0, f64::from(u32::try_from(slot + 1).unwrap()))
                .unwrap();
        }
        let original_ptr = initial.values.as_ptr();
        let mut reused = history
            .trial_in_reusing(IterationPhase::Float, None, Some(initial))
            .unwrap();
        assert_eq!(original_ptr, reused.values.as_ptr());
        assert!(reused.values().iter().all(|value| value.is_nan()));
        assert_eq!(reused.phase(), IterationPhase::Float);
        history
            .device(&mut reused, 1..2)
            .unwrap()
            .set(0, 9.)
            .unwrap();
        let error = history.check(&reused).unwrap_err();
        assert!(error.to_string().contains("slot 0 is unset"));
        for slot in [0, 2] {
            history
                .device(&mut reused, slot..slot + 1)
                .unwrap()
                .set(0, 2.)
                .unwrap();
        }
        history.check(&reused).unwrap();
        let again = history
            .trial_in_reusing(IterationPhase::Junction, None, Some(reused))
            .unwrap();
        assert_eq!(again.values.as_ptr(), original_ptr);
        assert!(again.values().iter().all(|value| value.is_nan()));
    }

    #[test]
    fn trial_pool_never_mutates_shared_snapshots_or_stale_revisions() {
        let mut history = StateHistory::new(2);
        let mut initial = history.trial();
        history
            .device(&mut initial, 0..1)
            .unwrap()
            .set(0, 17.)
            .unwrap();
        history
            .device(&mut initial, 1..2)
            .unwrap()
            .set(0, 19.)
            .unwrap();
        let held = Arc::clone(&initial.values);
        let trial = history
            .trial_in_reusing(IterationPhase::Float, None, Some(initial))
            .unwrap();
        assert_eq!(&*held, &[17., 19.]);
        assert_ne!(trial.values.as_ptr(), held.as_ptr());
        let stale = trial;
        let stale_ptr = stale.values.as_ptr();
        history.clear();
        assert!(history.check_trial(&stale).is_err());
        let fresh = history
            .trial_in_reusing(IterationPhase::Predict, None, Some(stale))
            .unwrap();
        assert_eq!(fresh.values.as_ptr(), stale_ptr);
        assert_eq!(fresh.phase(), IterationPhase::Predict);
        assert_eq!(fresh.values().len(), 2);
        assert!(fresh.values().iter().all(|value| value.is_nan()));
        assert!(history.check_trial(&fresh).is_ok());
        assert_eq!(&*held, &[17., 19.]);
    }

    #[test]
    fn invalid_commits_leave_the_history_unchanged() {
        let mut history = StateHistory::new(2);
        let mut trial = history.trial();
        history
            .device(&mut trial, 0..1)
            .unwrap()
            .set(0, 1.0)
            .unwrap();
        // Slot 1 is still unset.
        let error = history.commit(trial.clone()).unwrap_err();
        assert!(error.to_string().contains("slot 1 is unset"), "{error}");
        assert_eq!(history.depth(), 0);
        assert!(history.commit(StateHistory::new(3).trial()).is_err());
        assert_eq!(history.depth(), 0);
    }

    #[test]
    fn device_windows_are_bounded_and_finite() {
        let history = StateHistory::new(3);
        let mut trial = history.trial();
        assert!(history.device(&mut trial, 2..4).is_err());
        assert!(
            history
                .device(&mut trial, std::ops::Range { start: 2, end: 1 })
                .is_err()
        );
        let mut short = StateHistory::new(1).trial();
        assert!(history.device(&mut short, 0..1).is_err());
        let mut device = history.device(&mut trial, 1..3).unwrap();
        assert_eq!(device.len(), 2);
        assert!(device.trial(0).unwrap().is_nan());
        assert!(device.set(2, 1.0).is_err());
        assert!(device.set(0, f64::INFINITY).is_err());
        device.set(1, 5.0).unwrap();
        assert_eq!(trial.values()[2], 5.0);
        assert_eq!(trial.slice(2..3), Some(&[5.0][..]));
        let mut none = DeviceState::none();
        assert!(none.is_empty());
        assert_eq!(none.accepted(1, 0), None);
        assert_eq!(none.iterate(0), None);
        assert!(none.report_nonconvergence().is_err());
    }

    #[test]
    fn duplicate_delay_updates_never_commit_fixed_or_delay_state() {
        use crate::devices::delay::{DelayHistory, DelayUpdate, WaveformUpdate};

        let mut history = StateHistory::new(1).with_delay_lines([
            (3, DelayHistory::waveform(1)),
            (1000, DelayHistory::waveform(1)),
        ]);
        let mut trial = history.trial();
        history
            .device(&mut trial, 0..1)
            .unwrap()
            .set(0, 42.)
            .unwrap();
        let original = history.clone();
        let update = |time, breakpoint| {
            DelayUpdate::Waveform(WaveformUpdate {
                append: Some((time, vec![time])),
                breakpoints: vec![breakpoint],
                ..WaveformUpdate::default()
            })
        };
        let mut hook_ran = false;
        let error = history
            .accept_transaction(
                trial.clone(),
                super::HistoryPublication::Rotate,
                |_, _| Ok(vec![(3, update(1., 3.)), (3, update(2., 4.))]),
                |_| {
                    hook_ran = true;
                    Ok(())
                },
            )
            .unwrap_err();
        assert!(error.to_string().contains("duplicate delay update"));
        assert!(!hook_ran);
        assert_eq!(history, original);

        assert!(
            history
                .check_delays(&[(3, update(1., 3.)), (999, update(1., 4.))])
                .is_err()
        );
        assert_eq!(history, original);
        let updates = || vec![(1000, update(1., 4.)), (3, update(1., 3.))];
        assert!(
            history
                .accept_transaction(
                    trial.clone(),
                    super::HistoryPublication::Rotate,
                    |_, _| Ok(updates()),
                    |_| Err(super::state_error("hook refused")),
                )
                .is_err()
        );
        assert_eq!(history, original);
        let breaks = history
            .accept_transaction(
                trial,
                super::HistoryPublication::Rotate,
                |_, _| Ok(updates()),
                |_| Ok(()),
            )
            .unwrap();
        assert_eq!(breaks, vec![4., 3.]);
        assert!(history.delay(999).is_none());
        assert_eq!(history.accepted(1), Some(&[42.][..]));
        for device in [3, 1000] {
            assert_eq!(
                history
                    .delay(device)
                    .unwrap()
                    .as_waveform()
                    .unwrap()
                    .last_time(),
                Some(1.)
            );
        }
    }

    #[test]
    fn state1_patches_are_local_last_write_wins_and_transactional() {
        let mut history = StateHistory::new(4);
        for step in 0..4 {
            let mut trial = history.trial();
            let mut device = history.device(&mut trial, 0..4).unwrap();
            for slot in 0..4 {
                device.set(slot, f64::from(step)).unwrap();
            }
            history.commit(trial).unwrap();
        }
        let original = history.clone();
        let mut trial = history.trial();
        {
            let mut first = history.device(&mut trial, 0..2).unwrap();
            first.set(0, 7.).unwrap();
            first.set(1, 8.).unwrap();
        }
        {
            let mut second = history.device(&mut trial, 2..4).unwrap();
            second.patch_state1(0, 10.).unwrap();
            second.patch_state1(0, 11.).unwrap();
            second.patch_state1(1, 12.).unwrap();
            second.set(0, 9.).unwrap();
            second.set(1, 10.).unwrap();
            assert_eq!(second.accepted(1, 0), Some(11.));
            assert_eq!(second.accepted(1, usize::MAX), None);
        }
        {
            let first = history.device(&mut trial, 0..2).unwrap();
            assert_eq!(first.accepted(1, 0), Some(3.));
            assert_eq!(first.accepted(1, 2), None);
            assert_eq!(first.accepted(1, usize::MAX), None);
        }
        let mut next = history
            .trial_in(IterationPhase::Float, Some(&trial))
            .unwrap();
        assert!(std::sync::Arc::ptr_eq(
            next.previous.as_ref().unwrap(),
            &trial.values
        ));
        assert!(std::sync::Arc::ptr_eq(
            next.patches.0.as_ref().unwrap(),
            trial.patches.0.as_ref().unwrap()
        ));
        {
            let mut device = history.device(&mut next, 2..4).unwrap();
            assert_eq!(device.iterate(0), Some(9.));
            device.patch_state1(0, 20.).unwrap();
            assert_eq!(device.accepted(1, 0), Some(20.));
        }
        // Rejecting the second load leaves the first trial and original
        // history untouched. A cloned trial's value writes detach too.
        let mut rejected = trial.clone();
        {
            let mut device = history.device(&mut rejected, 2..4).unwrap();
            device.set(0, 99.).unwrap();
        }
        assert_eq!(trial.values()[2], 9.);
        assert_eq!(trial.patches.get(2), Some(11.));
        assert_eq!(history, original);
        history.commit(trial).unwrap();
        assert_eq!(history.accepted(1), Some(&[7., 8., 9., 10.][..]));
        for (age, value) in [(2, 3.), (3, 2.), (4, 1.)] {
            assert_eq!(history.accepted(age), Some(&[value, value, 11., 12.][..]));
        }
        assert_eq!(history.accepted(5), Some(&[0., 0., 0., 0.][..]));
        assert_eq!(original.accepted(1), Some(&[3., 3., 3., 3.][..]));
    }

    #[test]
    fn iterates_are_read_only_and_never_committed() {
        let mut history = StateHistory::new(2);
        let mut first = history.trial();
        assert_eq!(first.phase(), IterationPhase::Junction);
        {
            let mut device = history.device(&mut first, 0..2).unwrap();
            assert_eq!(device.iterate(0), None);
            device.set(0, 1.0).unwrap();
            device.set(1, 2.0).unwrap();
        }
        assert!(!first.is_nonconvergent());
        let mut second = history
            .trial_in(IterationPhase::Float, Some(&first))
            .unwrap();
        {
            let mut device = history.device(&mut second, 1..2).unwrap();
            assert_eq!(device.phase(), IterationPhase::Float);
            assert_eq!(device.iterate(0), Some(2.0));
            assert!(device.trial(0).unwrap().is_nan());
            device.report_nonconvergence().unwrap();
            device.set(0, 3.0).unwrap();
        }
        assert!(second.is_nonconvergent());
        history
            .device(&mut second, 0..1)
            .unwrap()
            .set(0, 4.0)
            .unwrap();
        history.commit(second).unwrap();
        assert_eq!(history.accepted(1), Some(&[4.0, 3.0][..]));
        assert!(
            history
                .trial_in(IterationPhase::Fix, Some(&StateHistory::new(3).trial()))
                .is_err()
        );
    }
}
