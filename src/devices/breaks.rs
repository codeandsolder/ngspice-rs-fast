//! The transient breakpoint table and the independent sources' accept-time
//! breakpoint scheduling.
//!
//! Upstream behavior: `src/spicelib/analysis/cktsetbk.c` (`CKTsetBreak`),
//! `cktclrbk.c` (`CKTclrBreak`), `src/spicelib/devices/vsrc/vsrcacct.c`
//! (`VSRCaccept`) and `isrc/isrcacct.c` (`ISRCaccept`). C does not precompute a
//! source's corners: at every accepted time point (`CKTaccept`) each source
//! requests at most its *next* corner, computed from the accepted time with
//! C's floating-point operation order, and `CKTsetBreak` merges the request
//! into a sorted table. The transient driver lands on (and equalises toward)
//! the table's first entry, so the exact breakpoint values decide the accepted
//! time points; this module reproduces them bit for bit.
//!
//! Like C (`vsrcacct.c`), SIN, EXP, SFFM and AM request no breakpoints; PULSE and
//! PWL do. Voltage and current PULSE sources schedule differently, as in C:
//! `VSRCaccept` requests the next corner after `time + CKTminBreak` whenever the
//! accepted time passed its last request, while `ISRCaccept` keeps the older
//! scheme that requests the following corner only when the accepted point is a
//! breakpoint (`CKTbreak`) lying within `1e-7 * PW` of a corner. The port's
//! [`Waveform::Step`] (no netlist spelling) schedules like a one-knot PWL.
//! ngspice's `xs` compatibility mode (PULSE phase in degrees) is not modelled,
//! matching [`crate::devices::Pulse`].

use super::pulse::cycle_position;
use crate::devices::{LinearSystem, SourceKind, TransientTiming, Waveform};
use crate::primitives::{Real, SpiceError, SpiceResult, almost_equal_ulps};

/// Numeric key within one ascending run of the table. Most of a transient
/// uses only the first run. C's two-entry clear can append the final-time
/// sentinel behind a later event; the second run represents that inversion
/// without sorting it away or reverting to a linear scan.
#[derive(Debug, Clone, Copy)]
struct EventKey {
    after_terminal: bool,
    time: Real,
}

impl EventKey {
    const fn new(time: Real, after_terminal: bool) -> Self {
        // C compares signed zeros equal. Preserve their original bits in the
        // bucket, while giving both the same numeric search key.
        Self {
            after_terminal,
            time: if time.to_bits() << 1 == 0 { 0. } else { time },
        }
    }
}

impl PartialEq for EventKey {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}
impl Eq for EventKey {}
impl PartialOrd for EventKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for EventKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        #[cfg(test)]
        tests::KEY_COMPARISONS.with(|count| {
            if let Some(n) = count.get() {
                count.set(Some(n + 1));
            }
        });
        self.after_terminal
            .cmp(&other.after_terminal)
            .then_with(|| self.time.total_cmp(&other.time))
    }
}

/// Equal numeric events in insertion order. The common singleton requires
/// no separate allocation. Duplicates preserve distinct signed-zero bits.
#[derive(Debug, Clone)]
struct Events {
    first: Real,
    rest: std::collections::VecDeque<Real>,
}

impl Events {
    const fn new(first: Real) -> Self {
        Self {
            first,
            rest: std::collections::VecDeque::new(),
        }
    }

    fn last(&self) -> Real {
        self.rest.back().copied().unwrap_or(self.first)
    }

    fn iter(&self) -> impl Iterator<Item = Real> + '_ {
        std::iter::once(self.first).chain(self.rest.iter().copied())
    }
}

/// C's breakpoint table `CKTbreaks`: at least two entries, normally ascending.
///
/// Lookup and removal use the ordered index rather than scanning or shifting
/// the pending events. Equal entries and the terminal sentinel retain C order.
#[derive(Debug, Clone)]
pub struct BreakTable {
    breaks: std::collections::BTreeMap<EventKey, Events>,
    len: usize,
    min_break: Real,
    final_time: Real,
}

impl PartialEq for BreakTable {
    fn eq(&self, other: &Self) -> bool {
        (self.min_break, self.final_time, self.len).eq(&(
            other.min_break,
            other.final_time,
            other.len,
        )) && self.times().eq(other.times())
    }
}

impl BreakTable {
    /// `dctran.c`'s initial table `{0, CKTfinalTime}` with merge distance
    /// `min_break` (`CKTminBreak`).
    /// # Errors
    /// Nonfinite or negative final time or merge distance.
    pub fn new(final_time: Real, min_break: Real) -> SpiceResult<Self> {
        if !(final_time.is_finite() && final_time >= 0. && min_break.is_finite() && min_break >= 0.)
        {
            return Err(SpiceError::circuit(
                "invalid final time or breakpoint merge distance",
            ));
        }
        let mut table = Self {
            breaks: std::collections::BTreeMap::new(),
            len: 0,
            min_break,
            final_time,
        };
        table.insert(0., false);
        table.insert(final_time, false);
        Ok(table)
    }

    /// `CKTminBreak`.
    #[must_use = "the returned interval determines breakpoint coalescing"]
    pub const fn min_break(&self) -> Real {
        self.min_break
    }

    /// `CKTbreaks[0]`, the next breakpoint.
    #[must_use = "the returned time bounds the next accepted step"]
    pub fn first(&self) -> Real {
        self.breaks
            .first_key_value()
            .map_or(self.final_time, |(_, events)| events.first)
    }

    /// `CKTbreaks[1]`, the breakpoint after the next.
    #[must_use = "the returned time identifies the following pending breakpoint"]
    pub fn second(&self) -> Real {
        self.times().nth(1).unwrap_or(self.final_time)
    }

    /// Borrow the event order (including duplicate sentinels); no snapshot
    /// allocation is needed to inspect or iterate the pending table.
    pub fn times(&self) -> impl Iterator<Item = Real> + '_ {
        self.breaks.values().flat_map(Events::iter)
    }

    /// The pending entry count, including both terminal sentinel entries.
    #[must_use = "the count includes the two required sentinel entries"]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// A valid breakpoint table always has at least two entries.
    #[must_use = "the result checks the invariant that pending sentinels remain"]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn insert(&mut self, time: Real, after_terminal: bool) {
        self.breaks
            .entry(EventKey::new(time, after_terminal))
            .and_modify(|events| events.rest.push_back(time))
            .or_insert_with(|| Events::new(time));
        self.len += 1;
    }

    fn remove_first_of(&mut self, key: EventKey) {
        if let std::collections::btree_map::Entry::Occupied(mut entry) = self.breaks.entry(key) {
            let events = entry.get_mut();
            if let Some(next) = events.rest.pop_front() {
                events.first = next;
            } else {
                entry.remove();
            }
            self.len -= 1;
        }
    }

    /// The first strictly greater entry in sequence order (not necessarily
    /// the globally smallest greater time after C's terminal inversion).
    fn successor(&self, time: Real) -> Option<(EventKey, Real)> {
        use std::ops::Bound::{Excluded, Included, Unbounded};
        self.breaks
            .range((
                Excluded(EventKey::new(time, false)),
                Included(EventKey::new(Real::INFINITY, false)),
            ))
            .next()
            .or_else(|| {
                self.breaks
                    .range((Excluded(EventKey::new(time, true)), Unbounded))
                    .next()
            })
            .map(|(&key, events)| (key, events.first))
    }

    /// `CKTsetBreak(ckt, time)` with `CKTtime = now`.
    ///
    /// A request within 3 ULPs of `now` is ignored. Preserve C's successor-
    /// first merge priority, even if the predecessor is an exact duplicate.
    /// # Errors
    /// `time` is in the past (C: "breakpoint in the past - HELP!") or not finite.
    pub fn set(&mut self, now: Real, time: Real) -> SpiceResult<()> {
        if almost_equal_ulps(time, now, 3) {
            return Ok(());
        }
        if now > time || !time.is_finite() {
            return Err(SpiceError::Numerical {
                context: "transient".to_owned(),
                message: format!("breakpoint {time:e} in the past (time {now:e})"),
            });
        }
        if let Some((key, successor)) = self.successor(time) {
            if successor - time <= self.min_break {
                // Replace exactly the first occurrence of the successor.
                self.remove_first_of(key);
                self.insert(time, key.after_terminal);
            } else if !self
                .breaks
                .range(..key)
                .next_back()
                .is_some_and(|(_, events)| time - events.last() <= self.min_break)
            {
                self.insert(time, key.after_terminal);
            }
        } else if let Some((&key, events)) = self.breaks.last_key_value()
            && time - events.last() > self.min_break
        {
            self.insert(time, key.after_terminal);
        }
        Ok(())
    }

    /// `CKTclrBreak`: drops the first entry; a two-entry table becomes
    /// `{CKTbreaks[1], CKTfinalTime}`, even if that reverses the order.
    pub fn clear(&mut self) {
        if let Some((&key, _)) = self.breaks.first_key_value() {
            self.remove_first_of(key);
        }
        if self.len == 1
            && let Some((&key, events)) = self.breaks.first_key_value()
        {
            let next = events.first;
            if next > self.final_time {
                // Re-establish at most one inversion. A lone event from the
                // second run becomes the first run before appending the
                // (smaller) sentinel.
                if key.after_terminal {
                    self.remove_first_of(key);
                    self.insert(next, false);
                }
                self.insert(self.final_time, true);
            } else {
                self.insert(self.final_time, key.after_terminal);
            }
        }
    }
}

/// Event state for a single source with a schedulable waveform.
/// Source indices retain their original `DEVaccept` ordering within each kind.
#[derive(Debug, Clone, Copy, PartialEq)]
struct ActiveSource {
    index: usize,
    break_time: Real,
}

/// A current PULSE only requests work when the driver landed on a break.
/// Unknown defaults remain active to preserve unresolved-input errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Eligibility {
    Inert,
    AtBreakOnly,
    EveryAccept,
}

fn eligibility(waveform: &Waveform) -> Eligibility {
    match waveform {
        Waveform::Constant(_) | Waveform::Function(_) => Eligibility::Inert,
        Waveform::PortSeries { baseline, .. } => eligibility(baseline),
        Waveform::Pulse(_) => Eligibility::AtBreakOnly,
        Waveform::Pwl(_)
        | Waveform::Step { .. }
        | Waveform::PulseDefaults(_)
        | Waveform::FunctionDefaults(_) => Eligibility::EveryAccept,
    }
}

/// Accept-time breakpoint scheduling bound to one immutable `LinearSystem`.
///
/// Owning a borrow prevents stale source indices after a system replacement.
/// Construct a new scheduler for another system or reset the current one.
#[derive(Debug, Clone)]
pub struct SourceBreaks<'a> {
    system: &'a LinearSystem,
    voltage: Vec<ActiveSource>,
    current: Vec<ActiveSource>,
    /// Positions in the ordered current list checked off-break too.
    current_always: Vec<usize>,
}

impl PartialEq for SourceBreaks<'_> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.system, other.system)
            && self.voltage == other.voltage
            && self.current == other.current
            && self.current_always == other.current_always
    }
}

impl<'a> SourceBreaks<'a> {
    /// Bind the event-capable source index once per immutable system.
    /// C initializes voltage break-time to -1 and current break-time to 0.
    #[must_use = "the returned index binds source event schedules to this immutable system"]
    pub fn new(system: &'a LinearSystem) -> Self {
        let mut voltage = Vec::new();
        let mut current = Vec::new();
        let mut current_always = Vec::new();
        for (index, source) in system.sources.iter().enumerate() {
            let eligibility = eligibility(&source.waveform);
            if eligibility == Eligibility::Inert {
                continue;
            }
            match source.kind {
                SourceKind::Voltage => voltage.push(ActiveSource {
                    index,
                    break_time: -1.,
                }),
                SourceKind::Current => {
                    if eligibility != Eligibility::AtBreakOnly {
                        current_always.push(current.len());
                    }
                    current.push(ActiveSource {
                        index,
                        break_time: 0.,
                    });
                }
            }
        }
        Self {
            system,
            voltage,
            current,
            current_always,
        }
    }

    /// Restart the phases on the same bound system without reallocating.
    pub fn reset(&mut self) {
        for source in &mut self.voltage {
            source.break_time = -1.;
        }
        for source in &mut self.current {
            source.break_time = 0.;
        }
    }

    /// Request all source breakpoints for one physical source kind.
    /// The driver visits current sources, then delay-device accept hooks,
    /// then voltage sources. Within each phase source order is unchanged.
    ///
    /// # Errors
    /// A request in the past, or unresolved PULSE/function defaults.
    pub fn accept(
        &mut self,
        kind: SourceKind,
        now: Real,
        at_break: bool,
        timing: &TransientTiming,
        table: &mut BreakTable,
    ) -> SpiceResult<()> {
        let request = Request {
            now,
            at_break,
            timing,
            kind,
        };
        match kind {
            SourceKind::Voltage => {
                for source in &mut self.voltage {
                    request.waveform(
                        &self.system.sources[source.index].waveform,
                        &mut source.break_time,
                        table,
                    )?;
                }
            }
            SourceKind::Current if at_break => {
                for source in &mut self.current {
                    request.waveform(
                        &self.system.sources[source.index].waveform,
                        &mut source.break_time,
                        table,
                    )?;
                }
            }
            SourceKind::Current => {
                for &position in &self.current_always {
                    let source = &mut self.current[position];
                    request.waveform(
                        &self.system.sources[source.index].waveform,
                        &mut source.break_time,
                        table,
                    )?;
                }
            }
        }
        Ok(())
    }
}

struct Request<'a> {
    now: Real,
    at_break: bool,
    timing: &'a TransientTiming,
    kind: SourceKind,
}

impl Request<'_> {
    fn waveform(
        &self,
        waveform: &Waveform,
        break_time: &mut Real,
        table: &mut BreakTable,
    ) -> SpiceResult<()> {
        match waveform {
            Waveform::Constant(_) | Waveform::Function(_) => Ok(()),
            Waveform::PortSeries { baseline, .. } => self.waveform(baseline, break_time, table),
            Waveform::Pulse(pulse) => match self.kind {
                SourceKind::Voltage => self.vsrc_pulse(pulse, break_time, table),
                SourceKind::Current => self.isrc_pulse(pulse, table),
            },
            Waveform::Pwl(pwl) => {
                self.pwl(pwl.knots(), pwl.delay(), pwl.repeat(), break_time, table)
            }
            Waveform::Step { time, .. } => self.pwl(&[(*time, 0.)], 0., None, break_time, table),
            Waveform::PulseDefaults(_) | Waveform::FunctionDefaults(_) => Err(SpiceError::circuit(
                "source defaults are unresolved; bind transient timing before scheduling breakpoints",
            )),
        }
    }

    /// `VSRCaccept`, `case PULSE` (default compatibility mode).
    fn vsrc_pulse(
        &self,
        pulse: &crate::devices::Pulse,
        break_time: &mut Real,
        table: &mut BreakTable,
    ) -> SpiceResult<()> {
        let now = self.now;
        let (td, tr, tf, pw, per) = (
            pulse.delay(),
            pulse.rise(),
            pulse.fall(),
            pulse.width(),
            pulse.period(),
        );
        // Offset time by delay.
        let mut time = now - td;
        if let Some(tmax) = pulse.stop()
            && time > tmax
        {
            return Ok(());
        }
        if now >= *break_time {
            if time >= per {
                // Repeating signal: where in period are we?
                time = cycle_position(time, per).phase;
            }
            // A request very close to the current time would be ignored:
            // adjust so the next corner is selected.
            let min_break = table.min_break();
            let atime = time + min_break;
            let wait = if atime < 0. {
                -time
            } else if atime < tr {
                tr - time
            } else if atime < tr + pw {
                tr + pw - time
            } else if atime < tr + pw + tf {
                tr + pw + tf - time
            } else {
                per - time
            };
            *break_time = now + wait;
            table.set(now, *break_time)?;
            // A timestep ending just before the break time may have the
            // request ignored: lower the threshold for the following one.
            *break_time -= min_break;
        }
        Ok(())
    }

    /// `ISRCaccept`, `case PULSE` (default compatibility mode): the
    /// `CKTbreak`/`SAMETIME` scheme.
    fn isrc_pulse(&self, pulse: &crate::devices::Pulse, table: &mut BreakTable) -> SpiceResult<()> {
        let now = self.now;
        let (td, tr, tf, pw, per) = (
            pulse.delay(),
            pulse.rise(),
            pulse.fall(),
            pulse.width(),
            pulse.period(),
        );
        // #define SAMETIME(a,b) (fabs((a)-(b)) <= TIMETOL * PW), TIMETOL 1e-7
        let same = |a: Real, b: Real| (a - b).abs() <= 1e-7 * pw;
        let mut time = now - td;
        let tshift = td;
        let mut basetime = 0.;
        if let Some(tmax) = pulse.stop()
            && time > tmax
        {
            return Ok(());
        }
        if time >= per {
            let cycle = cycle_position(time, per);
            basetime = cycle.base;
            time = cycle.phase;
        }
        if !self.at_break {
            return Ok(());
        }
        let request = if time <= 0. || time >= tr + pw + tf {
            if same(time, 0.) {
                Some(basetime + tr + tshift)
            } else if same(tr + pw + tf, time) {
                Some(basetime + per + tshift)
            } else if time.partial_cmp(&-tshift) == Some(std::cmp::Ordering::Equal) {
                // C's exact delay boundary; signed zero compares equal.
                Some(basetime + tshift)
            } else if same(per, time) {
                Some(basetime + tshift + tr + per)
            } else {
                None
            }
        } else if time >= tr && time <= tr + pw {
            if same(time, tr) {
                Some(basetime + tshift + tr + pw)
            } else if same(tr + pw, time) {
                Some(basetime + tshift + tr + pw + tf)
            } else {
                None
            }
        } else if time > 0. && time < tr {
            if same(time, 0.) {
                Some(basetime + tshift + tr)
            } else if same(time, tr) {
                Some(basetime + tshift + tr + pw)
            } else {
                None
            }
        } else if same(time, tr + pw) {
            // time > TR + PW && < TR + PW + TF
            Some(basetime + tshift + tr + pw + tf)
        } else if same(time, tr + pw + tf) {
            Some(basetime + tshift + per)
        } else {
            None
        };
        request.map_or(Ok(()), |time| table.set(now, time))
    }

    /// `VSRCaccept`/`ISRCaccept`, `case PWL`: borrowed knots (as given),
    /// `td=` delay and the `r=` restart knot index.
    fn pwl(
        &self,
        knots: &[(Real, Real)],
        delay: Real,
        repeat: Option<usize>,
        break_time: &mut Real,
        table: &mut BreakTable,
    ) -> SpiceResult<()> {
        let now = self.now;
        if now < *break_time {
            return Ok(());
        }
        let mut time = now - delay;
        let Some(&(end, _)) = knots.last() else {
            return Ok(());
        };
        if time > end {
            if let Some(index) = repeat {
                let start = knots[index].0;
                let period = end - start;
                time -= start;
                time = cycle_position(time, period).phase + start;
            } else {
                *break_time = self.timing.final_time();
                return Ok(());
            }
        }
        let min_break = table.min_break();
        let atime = time + min_break;
        if atime.is_nan() {
            return Ok(());
        }
        let next = knots.partition_point(|(knot, _)| *knot <= atime);
        if let Some(&(knot, _)) = knots.get(next) {
            *break_time = now + knot - time;
            table.set(now, *break_time)?;
            *break_time -= min_break;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::devices::Pulse;
    use crate::devices::linear::LinearSource;
    use crate::primitives::Complex;

    fn schedulable_source(kind: SourceKind, waveform: Waveform) -> LinearSource {
        LinearSource {
            name: std::sync::Arc::new("schedule-test".to_owned()),
            kind,
            rows: crate::devices::linear::SourceRows::new(),
            dc: 0.,
            ac: Complex::ZERO,
            waveform: waveform.into(),
        }
    }

    /// The pre-#136 accept loop: two full passes, with the original
    /// initialization and no cached eligibility. Its Request kernel is
    /// shared so the differential compares source visits, not arithmetic.
    fn reference_accept(
        system: &LinearSystem,
        times: &mut [Real],
        kind: SourceKind,
        now: Real,
        at_break: bool,
        timing: &TransientTiming,
        table: &mut BreakTable,
    ) -> SpiceResult<()> {
        for (source, break_time) in system.sources.iter().zip(times) {
            if source.kind != kind {
                continue;
            }
            Request {
                now,
                at_break,
                timing,
                kind,
            }
            .waveform(&source.waveform, break_time, table)?;
        }
        Ok(())
    }

    #[test]
    fn active_source_schedule_matches_full_c_order_reference_and_rebinds() {
        let mut system = LinearSystem::new(0);
        for index in 0..128 {
            system.sources.push(schedulable_source(
                if index % 2 == 0 {
                    SourceKind::Voltage
                } else {
                    SourceKind::Current
                },
                Waveform::Constant(0.),
            ));
        }
        let pulse = Waveform::Pulse(Pulse::new(0., 1., 0., 0.1, 0.1, 0.4, 2.).unwrap());
        system
            .sources
            .push(schedulable_source(SourceKind::Voltage, pulse.clone()));
        system
            .sources
            .push(schedulable_source(SourceKind::Current, pulse));
        system.sources.push(schedulable_source(
            SourceKind::Voltage,
            Waveform::Step {
                before: 0.,
                after: 1.,
                time: 0.5,
            },
        ));
        system.sources.push(schedulable_source(
            SourceKind::Current,
            Waveform::PortSeries {
                baseline: Box::new(Waveform::Step {
                    before: 0.,
                    after: 1.,
                    time: 1.5,
                }),
                tones: vec![],
            },
        ));
        let timing = TransientTiming::new(0.05, 10.).unwrap();
        let mut active = SourceBreaks::new(&system);
        assert_eq!(active.voltage.len(), 2);
        assert_eq!(active.current.len(), 2);
        assert_eq!(active.current_always, [1]);
        let initial = active.clone();
        let mut time_state = system
            .sources
            .iter()
            .map(|s| match s.kind {
                SourceKind::Voltage => -1.,
                SourceKind::Current => 0.,
            })
            .collect::<Vec<_>>();
        let mut indexed = BreakTable::new(10., 1e-10).unwrap();
        let mut reference = indexed.clone();
        for (now, at_break) in [
            (0., true),
            (0.05, false),
            (0.1, true),
            (0.3, false),
            (0.5, true),
            (0.6, false),
            (1., true),
            (1.5, true),
            (2., true),
            (2.4, false),
        ] {
            for kind in [SourceKind::Current, SourceKind::Voltage] {
                let old = reference_accept(
                    &system,
                    &mut time_state,
                    kind,
                    now,
                    at_break,
                    &timing,
                    &mut reference,
                );
                let new = active.accept(kind, now, at_break, &timing, &mut indexed);
                assert_eq!(
                    old.as_ref().err().map(ToString::to_string),
                    new.as_ref().err().map(ToString::to_string),
                    "kind={kind:?} time={now} break={at_break}"
                );
                assert!(
                    indexed
                        .times()
                        .map(Real::to_bits)
                        .eq(reference.times().map(Real::to_bits)),
                    "event table differs at {now} for {kind:?}"
                );
                if old.is_err() {
                    break;
                }
            }
        }
        active.reset();
        assert_eq!(active, initial);
        let second = system.clone();
        let rebound = SourceBreaks::new(&second);
        assert_ne!(active, rebound, "bound identity must be part of equality");
    }

    #[test]
    #[ignore = "opt-in 1k/10k/100k source-scheduler benchmark"]
    fn active_scheduler_scales_with_active_not_total_sources() {
        let timing = TransientTiming::new(0.1, 10.).unwrap();
        for count in [1_000usize, 10_000, 100_000] {
            let mut system = LinearSystem::new(0);
            system.sources.reserve(count + 4);
            for n in 0..count {
                system.sources.push(schedulable_source(
                    if n % 2 == 0 {
                        SourceKind::Voltage
                    } else {
                        SourceKind::Current
                    },
                    Waveform::Constant(0.),
                ));
            }
            for kind in [SourceKind::Current, SourceKind::Voltage] {
                system.sources.push(schedulable_source(
                    kind,
                    Waveform::Step {
                        before: 0.,
                        after: 1.,
                        time: 5.,
                    },
                ));
            }
            let mut schedule = SourceBreaks::new(&system);
            assert_eq!(schedule.current.len() + schedule.voltage.len(), 2);
            let mut time_state = system
                .sources
                .iter()
                .map(|s| match s.kind {
                    SourceKind::Voltage => -1.,
                    SourceKind::Current => 0.,
                })
                .collect::<Vec<_>>();
            let mut old = BreakTable::new(10., 1e-12).unwrap();
            let mut new = old.clone();
            let start = std::time::Instant::now();
            for _ in 0..2000 {
                for kind in [SourceKind::Current, SourceKind::Voltage] {
                    reference_accept(&system, &mut time_state, kind, 0., false, &timing, &mut old)
                        .unwrap();
                }
            }
            let scan = start.elapsed();
            let start = std::time::Instant::now();
            for _ in 0..2000 {
                for kind in [SourceKind::Current, SourceKind::Voltage] {
                    schedule.accept(kind, 0., false, &timing, &mut new).unwrap();
                }
            }
            let bound = start.elapsed();
            assert!(
                old.times()
                    .map(Real::to_bits)
                    .eq(new.times().map(Real::to_bits))
            );
            println!(
                "sources={count} active=2 visits: old={} new={} scan={scan:?} bound={bound:?}",
                count * 4000,
                4000
            );
        }
    }

    std::thread_local! {
        pub(super) static KEY_COMPARISONS: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
    }

    /// The former C-order vector implementation, including the terminal
    /// inversion and exact comparisons. Independent oracle, not production.
    struct Reference {
        times: Vec<Real>,
        final_time: Real,
        min_break: Real,
        visits: usize,
        shifts: usize,
    }

    impl Reference {
        fn new(final_time: Real, min_break: Real) -> Self {
            Self {
                times: vec![0., final_time],
                final_time,
                min_break,
                visits: 0,
                shifts: 0,
            }
        }

        fn set(&mut self, now: Real, time: Real) -> bool {
            if almost_equal_ulps(time, now, 3) {
                return true;
            }
            if now > time || !time.is_finite() {
                return false;
            }
            for i in 0..self.times.len() {
                self.visits += 1;
                if self.times[i] > time {
                    if self.times[i] - time <= self.min_break {
                        self.times[i] = time;
                        return true;
                    }
                    if i > 0 && time - self.times[i - 1] <= self.min_break {
                        return true;
                    }
                    self.shifts += self.times.len() - i;
                    self.times.insert(i, time);
                    return true;
                }
            }
            if self
                .times
                .last()
                .is_none_or(|last| time - last > self.min_break)
            {
                self.times.push(time);
            }
            true
        }

        fn clear(&mut self) {
            if self.times.len() > 2 {
                self.shifts += self.times.len() - 1;
                self.times.remove(0);
            } else {
                self.times[0] = self.times[1];
                self.times[1] = self.final_time;
            }
        }

        fn compare(&self, table: &BreakTable) {
            assert_eq!(table.len(), self.times.len());
            assert_eq!(table.first().to_bits(), self.times[0].to_bits());
            assert_eq!(table.second().to_bits(), self.times[1].to_bits());
            assert!(
                table
                    .times()
                    .map(Real::to_bits)
                    .eq(self.times.iter().map(|x| x.to_bits())),
                "indexed: {:?}; reference: {:?}",
                table.times().collect::<Vec<_>>(),
                self.times
            );
        }
    }

    #[test]
    fn successor_priority_can_create_duplicates_and_preserves_zero_sign() {
        let mut table = BreakTable::new(10., 0.25).unwrap();
        for time in [1., 1.5, 1.25, 1.] {
            table.set(-1., time).unwrap();
        }
        assert_eq!(table.times().collect::<Vec<_>>(), [0., 1., 1., 10.]);

        let mut table = BreakTable::new(0.25, 0.25).unwrap();
        table.set(-1., -0.).unwrap();
        assert_eq!(
            table.times().map(Real::to_bits).collect::<Vec<_>>(),
            [0., -0.].map(Real::to_bits)
        );
        table.clear();
        table.set(-1., 0.).unwrap();
        assert_eq!(
            table.times().map(Real::to_bits).collect::<Vec<_>>(),
            [-0., 0.].map(Real::to_bits)
        );
        table.clear();
        assert_eq!(table.first().to_bits(), Real::to_bits(0.));
        assert_eq!(table.second().to_bits(), Real::to_bits(0.25));
    }

    #[test]
    fn inversion_after_stop_is_not_reordered_and_can_accept_more_events() {
        let mut table = BreakTable::new(10., 0.1).unwrap();
        let mut reference = Reference::new(10., 0.1);
        table.set(0., 12.).unwrap();
        assert!(reference.set(0., 12.));
        for _ in 0..2 {
            table.clear();
            reference.clear();
        }
        assert_eq!(table.times().collect::<Vec<_>>(), [12., 10.]);
        for time in [11., 13., 12.5, 12.45, 9., 10., 15., 12.5] {
            assert_eq!(table.set(0., time).is_ok(), reference.set(0., time));
            reference.compare(&table);
        }
        while reference.times.len() > 2 {
            table.clear();
            reference.clear();
            reference.compare(&table);
        }
        for _ in 0..5 {
            table.clear();
            reference.clear();
            reference.compare(&table);
        }
    }

    #[test]
    fn indexed_table_matches_every_randomized_reference_transition() {
        let mut random = 0x23a6_fce0_8125_c4d9_u64;
        let mut next = || {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            random
        };
        for final_time in [-0., 2., 10.] {
            for min_break in [0., Real::from_bits(1), 0.125, 0.5, 1.] {
                let mut table = BreakTable::new(final_time, min_break).unwrap();
                let mut reference = Reference::new(final_time, min_break);
                for _ in 0..5000 {
                    let bits = next();
                    if bits % 4 == 0 {
                        table.clear();
                        reference.clear();
                    } else {
                        let time = match bits % 19 {
                            0 => 0.,
                            1 => -0.,
                            2 => Real::from_bits(bits % 8),
                            3 => Real::INFINITY,
                            4 => Real::NEG_INFINITY,
                            5 => Real::NAN,
                            6 => Real::MAX,
                            7 => -Real::MAX,
                            8 => table.first(),
                            9 => table.second(),
                            _ => Real::from(u32::try_from(next() % 161).unwrap()) / 8. - 5.,
                        };
                        let now = match next() % 6 {
                            0 => time,
                            1 => Real::from_bits(time.to_bits().wrapping_add(2)),
                            2 => 0.,
                            3 => Real::from(u32::try_from(next() % 81).unwrap()) / 8. - 5.,
                            _ => -Real::MAX,
                        };
                        assert_eq!(
                            table.set(now, time).is_ok(),
                            reference.set(now, time),
                            "now={now:e} time={time:e}"
                        );
                    }
                    reference.compare(&table);
                }
            }
        }
    }

    #[test]
    fn invalid_constructor_inputs_cannot_enter_the_ordered_index() {
        for invalid in [Real::NAN, Real::INFINITY, Real::NEG_INFINITY, -1.] {
            assert!(BreakTable::new(invalid, 0.).is_err());
            assert!(BreakTable::new(1., invalid).is_err());
        }
    }

    #[test]
    #[ignore = "manual breakpoint lookup comparison count"]
    fn indexed_breakpoint_work() {
        for n in [4096_u32, 16_384, 65_536] {
            let finish = Real::from(n) + 1.;
            let mut table = BreakTable::new(finish, 0.).unwrap();
            let mut reference = Reference::new(finish, 0.);
            KEY_COMPARISONS.with(|count| count.set(Some(0)));
            // Multiplication by an odd number permutes the power-of-two range.
            for i in 0..n {
                let time = Real::from((i.wrapping_mul(40503) & (n - 1)) + 1);
                table.set(0., time).unwrap();
                assert!(reference.set(0., time));
            }
            reference.compare(&table);
            for _ in 0..n {
                table.clear();
                reference.clear();
            }
            reference.compare(&table);
            let comparisons = KEY_COMPARISONS.with(|count| count.replace(None).unwrap());
            println!(
                "work,n={n},reference_visits={},reference_shifts={},tree_key_comparisons={comparisons}",
                reference.visits, reference.shifts
            );
        }
    }

    #[test]
    fn set_break_merges_like_cktsetbk() {
        let mut table = BreakTable::new(10., 0.1).unwrap();
        table.set(0., 5.).unwrap();
        assert_eq!(table.times().collect::<Vec<_>>(), [0., 5., 10.]);
        // Within min_break before an entry: the earlier time replaces it.
        table.set(0., 4.95).unwrap();
        assert_eq!(table.times().collect::<Vec<_>>(), [0., 4.95, 10.]);
        // Within min_break after an entry: skipped.
        table.set(0., 5.0).unwrap();
        assert_eq!(table.times().collect::<Vec<_>>(), [0., 4.95, 10.]);
        // Beyond the end, close to the last entry: skipped; farther: appended.
        table.set(0., 10.05).unwrap();
        assert_eq!(table.times().collect::<Vec<_>>(), [0., 4.95, 10.]);
        table.set(0., 12.).unwrap();
        assert_eq!(table.times().collect::<Vec<_>>(), [0., 4.95, 10., 12.]);
        // A request at the current time is ignored, one in the past fails.
        table.set(4., 4.).unwrap();
        assert_eq!(table.len(), 4);
        assert!(table.set(4., 3.).is_err());
    }

    #[test]
    fn clear_break_keeps_two_entries() {
        let mut table = BreakTable::new(10., 0.1).unwrap();
        table.set(0., 5.).unwrap();
        table.clear();
        assert_eq!(table.times().collect::<Vec<_>>(), [5., 10.]);
        table.clear();
        assert_eq!(table.times().collect::<Vec<_>>(), [10., 10.]);
    }
}
