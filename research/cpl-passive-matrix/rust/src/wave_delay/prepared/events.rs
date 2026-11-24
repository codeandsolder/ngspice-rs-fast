//! Causally propagate physically distinct accepted-wave and source slope
//! breakpoints through the modal-delay / static-scattering graph.
//!
//! Reordered paths that have traversed exactly the same number of
//! half-delay units in each mode have the **same physical arrival**.
//! Encode them canonically as a source timestamp plus an integer
//! count vector of exact per-mode half delays. This avoids floating
//! point path-order drift and eliminates exponential redundant
//! propagation of delay permutations without losing any distinct
//! physical wavefront or waveform-corner event.
//!
//! A physical multimode network still has genuinely distinct sums
//! of incommensurate delays, so complete exact event enumeration
//! can require substantial work for long intervals. No events are
//! silently dropped or grouped merely by a timing tolerance.

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};

use super::super::{Trace, WaveNetwork};

/// The research wave network supports 1..32 modal paths; unlike a
/// packed integer encoding, this does not limit the number of steps
/// through any individual mode to a few bits.
const MAX_CANONICAL_MODES: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CanonicalPath {
    node: usize,
    root_time_bits: u64,
    half_delay_counts: [u32; MAX_CANONICAL_MODES],
}

impl CanonicalPath {
    const fn root(node: usize, time: f64, mode: usize, half_steps: u32) -> Self {
        let mut counts = [0; MAX_CANONICAL_MODES];
        counts[mode] = half_steps;
        Self {
            node,
            root_time_bits: time.to_bits(),
            half_delay_counts: counts,
        }
    }

    fn advanced(
        self,
        next_node: usize,
        mode: usize,
        additional_half_steps: u32,
    ) -> Result<Self, &'static str> {
        let mut next = self;
        next.node = next_node;
        next.half_delay_counts[mode] = next.half_delay_counts[mode]
            .checked_add(additional_half_steps)
            .ok_or("modal event crossed more delay links than u32 can count")?;
        Ok(next)
    }

    fn arrival(self, half_delays: &[f64]) -> f64 {
        let mut time = f64::from_bits(self.root_time_bits);
        for (count, &delay) in self.half_delay_counts.iter().zip(half_delays) {
            time = f64::from(*count).mul_add(delay, time);
        }
        time
    }
}

/// Heap ordering includes canonical path identity for deterministic
/// traversal when several distinct paths share one rounded arrival.
#[derive(Debug, Clone, Copy)]
struct Event {
    time: f64,
    path: CanonicalPath,
}
impl PartialEq for Event {
    fn eq(&self, other: &Self) -> bool {
        self.time.to_bits() == other.time.to_bits() && self.path == other.path
    }
}
impl Eq for Event {}
impl PartialOrd for Event {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Event {
    fn cmp(&self, other: &Self) -> Ordering {
        self.time
            .total_cmp(&other.time)
            .then_with(|| self.path.node.cmp(&other.path.node))
            .then_with(|| self.path.root_time_bits.cmp(&other.path.root_time_bits))
            .then_with(|| {
                self.path
                    .half_delay_counts
                    .cmp(&other.path.half_delay_counts)
            })
    }
}

fn offer(
    heap: &mut BinaryHeap<Reverse<Event>>,
    seen: &mut HashSet<CanonicalPath>,
    path: CanonicalPath,
    half_delays: &[f64],
    start: f64,
    finish: f64,
) -> Result<(), &'static str> {
    let time = path.arrival(half_delays);
    if time <= start || time > finish || !time.is_finite() || seen.contains(&path) {
        return Ok(());
    }
    seen.try_reserve(1)
        .map_err(|_| "coupled modal event identity allocation failed")?;
    heap.try_reserve(1)
        .map_err(|_| "coupled modal event queue allocation failed")?;
    seen.insert(path);
    heap.push(Reverse(Event { time, path }));
    Ok(())
}

fn seeds(trace: &Trace) -> impl Iterator<Item = f64> + '_ {
    trace
        .times
        .iter()
        .copied()
        .chain(trace.first_possible.iter().copied())
        .filter(|t| t.is_finite())
}

fn seed_boundaries(
    net: &WaveNetwork,
    finish: f64,
    heap: &mut BinaryHeap<Reverse<Event>>,
    seen: &mut HashSet<CanonicalPath>,
) -> Result<(), &'static str> {
    let n = net.modes;
    let width = 2 * n;
    for mode in 0..n {
        for root_time in seeds(&net.input_near) {
            for row in 0..width {
                if net.static_scattering[row * width + mode] != 0. {
                    offer(
                        heap,
                        seen,
                        CanonicalPath::root(row, root_time, mode, 1),
                        &net.half_delays,
                        net.time,
                        finish,
                    )?;
                }
            }
        }
        for root_time in seeds(&net.input_far) {
            let col = n + mode;
            for row in 0..width {
                if net.static_scattering[row * width + col] != 0. {
                    offer(
                        heap,
                        seen,
                        CanonicalPath::root((net.stages - 1) * width + row, root_time, mode, 1),
                        &net.half_delays,
                        net.time,
                        finish,
                    )?;
                }
            }
        }
    }
    Ok(())
}

fn seed_interior(
    net: &WaveNetwork,
    finish: f64,
    heap: &mut BinaryHeap<Reverse<Event>>,
    seen: &mut HashSet<CanonicalPath>,
) -> Result<(), &'static str> {
    let n = net.modes;
    let width = 2 * n;
    for stage in 0..net.stages {
        for mode in 0..n {
            let predecessors = [
                if stage > 0 {
                    Some((n + mode, &net.output_near[stage], (stage - 1) * width))
                } else {
                    None
                },
                if stage + 1 < net.stages {
                    Some((mode, &net.output_far[stage], (stage + 1) * width))
                } else {
                    None
                },
            ];
            for (input, trace, receiver) in predecessors.into_iter().flatten() {
                for root_time in seeds(trace) {
                    for row in 0..width {
                        if net.static_scattering[row * width + input] != 0. {
                            offer(
                                heap,
                                seen,
                                CanonicalPath::root(receiver + row, root_time, mode, 2),
                                &net.half_delays,
                                net.time,
                                finish,
                            )?;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn propagate_event(
    net: &WaveNetwork,
    event: Event,
    finish: f64,
    heap: &mut BinaryHeap<Reverse<Event>>,
    seen: &mut HashSet<CanonicalPath>,
    output_times: &mut Vec<f64>,
) -> Result<(), &'static str> {
    let modes = net.modes;
    let width = 2 * modes;
    let path = event.path;
    let stage = path.node / width;
    let port = path.node % width;
    let mode = port % modes;
    let receiver = if port < modes {
        if stage == 0 {
            None
        } else {
            Some(((stage - 1) * width, modes + mode))
        }
    } else if stage + 1 == net.stages {
        None
    } else {
        Some(((stage + 1) * width, mode))
    };
    if let Some((offset, input)) = receiver {
        for row in 0..width {
            if net.static_scattering[row * width + input] != 0. {
                offer(
                    heap,
                    seen,
                    path.advanced(offset + row, mode, 2)?,
                    &net.half_delays,
                    net.time,
                    finish,
                )?;
            }
        }
    } else {
        let external = path.advanced(path.node, mode, 1)?.arrival(&net.half_delays);
        if external > net.time && external <= finish {
            output_times.push(external);
        }
    }
    Ok(())
}

/// Return all physically distinguishable delayed interior/exterior
/// breakpoint times in the proposed macro interval. Distinct
/// propagation-path *orders* with the same modal-delay count vector
/// are deduplicated exactly, not by guessing a numerical tolerance.
///
/// # Errors
/// Exhausted modal path counter or allocation failure.
pub(super) fn propagated_breakpoints(
    net: &WaveNetwork,
    finish: f64,
) -> Result<Vec<f64>, &'static str> {
    let mut seen = HashSet::new();
    let mut heap = BinaryHeap::new();
    let mut events = Vec::new();
    seed_boundaries(net, finish, &mut heap, &mut seen)?;
    seed_interior(net, finish, &mut heap, &mut seen)?;
    while let Some(Reverse(event)) = heap.pop() {
        events
            .try_reserve(1)
            .map_err(|_| "coupled modal breakpoint list allocation failed")?;
        events.push(event.time);
        propagate_event(net, event, finish, &mut heap, &mut seen, &mut events)?;
    }
    events.sort_by(f64::total_cmp);
    events.dedup_by(|a, b| a.to_bits() == b.to_bits());
    Ok(events)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_path_identity_is_invariant_to_mode_delay_permutations() -> Result<(), &'static str>
    {
        let start = CanonicalPath::root(0, 0., 0, 1);
        let first = start.advanced(3, 1, 2)?.advanced(4, 2, 2)?;
        let second = start.advanced(3, 2, 2)?.advanced(4, 1, 2)?;
        assert_eq!(first, second);
        let delays = [1e-10, 1.414_213_562_373_095e-10, 1.732_050_807_568_877e-10];
        assert_eq!(
            first.arrival(&delays).to_bits(),
            second.arrival(&delays).to_bits()
        );
        let independent = second.advanced(4, 1, 1)?;
        assert_ne!(first, independent);
        assert_ne!(
            first.arrival(&delays).to_bits(),
            independent.arrival(&delays).to_bits()
        );
        Ok(())
    }

    #[test]
    fn canonical_path_rejects_counter_overflow_instead_of_wraparound() {
        let start = CanonicalPath::root(0, 0., 0, u32::MAX);
        assert!(start.advanced(1, 0, 1).is_err());
        assert!(start.advanced(1, 1, 1).is_ok());
    }
}
