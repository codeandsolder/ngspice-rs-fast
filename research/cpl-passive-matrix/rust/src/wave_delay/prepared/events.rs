//! Causal propagation of actual accepted-wave and external-launch slope
//! breakpoints through the modal delay and static-scattering graph.
//!
//! The regular method-of-steps grid limits interpolation error, but it
//! must also include real delayed corners: otherwise a segment crossing
//! an exact arrival can incorrectly reflow a slope discontinuity.

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};

use super::super::{Trace, WaveNetwork};

#[derive(Clone, Copy, Debug)]
struct Event {
    time: f64,
    node: usize,
}
impl PartialEq for Event {
    fn eq(&self, other: &Self) -> bool {
        self.time.to_bits() == other.time.to_bits() && self.node == other.node
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
            .then_with(|| self.node.cmp(&other.node))
    }
}

fn offer(
    heap: &mut BinaryHeap<Reverse<Event>>,
    seen: &mut HashSet<(usize, u64)>,
    node: usize,
    time: f64,
    start: f64,
    finish: f64,
) {
    if time > start && time <= finish && time.is_finite() && seen.insert((node, time.to_bits())) {
        heap.push(Reverse(Event { time, node }));
    }
}

fn seed_internal(
    network: &WaveNetwork,
    finish: f64,
    heap: &mut BinaryHeap<Reverse<Event>>,
    seen: &mut HashSet<(usize, u64)>,
) {
    let modes = network.modes;
    let width = 2 * modes;
    for stage in 0..network.stages {
        for mode in 0..modes {
            let neighbors = [
                if stage > 0 {
                    Some((
                        modes + mode,
                        &network.output_near[stage],
                        (stage - 1) * width,
                    ))
                } else {
                    None
                },
                if stage + 1 < network.stages {
                    Some((mode, &network.output_far[stage], (stage + 1) * width))
                } else {
                    None
                },
            ];
            for (input, trace, receiver) in neighbors.into_iter().flatten() {
                for time in trace
                    .times
                    .iter()
                    .copied()
                    .chain(trace.first_possible.iter().copied())
                {
                    let arrives = 2.0_f64.mul_add(network.half_delays[mode], time);
                    for row in 0..width {
                        if network.static_scattering[row * width + input] != 0. {
                            offer(heap, seen, receiver + row, arrives, network.time, finish);
                        }
                    }
                }
            }
        }
    }
}

/// Causally scheduled state or output slope breakpoints in a proposed
/// macro timestep. All nonzero static scatter entries represent possible
/// instantaneous cross-mode mixing; physical modal links provide strictly
/// positive edge delays.
pub(super) fn propagated_breakpoints(network: &WaveNetwork, finish: f64) -> Vec<f64> {
    let start = network.time;
    let modes = network.modes;
    let stages = network.stages;
    let width = 2 * modes;
    let mut seen = HashSet::new();
    let mut heap = BinaryHeap::new();
    let mut output_times = Vec::new();
    let scatter = &network.static_scattering;
    let half = &network.half_delays;

    // Every stored accepted knot (including the global start) may
    // change the slope of a retarded wave arriving after the start.
    // The physical wavefront is also a breakpoint, even without an
    // already accepted sample at that exact time.
    let seeds = |trace: &Trace| {
        trace
            .times
            .iter()
            .copied()
            .chain(trace.first_possible.iter().copied())
            .collect::<Vec<_>>()
    };
    for mode in 0..modes {
        for t in seeds(&network.input_near) {
            let arrives = t + half[mode];
            for row in 0..width {
                if scatter[row * width + mode] != 0. {
                    offer(&mut heap, &mut seen, row, arrives, start, finish);
                }
            }
        }
        for t in seeds(&network.input_far) {
            let arrives = t + half[mode];
            let col = modes + mode;
            for row in 0..width {
                if scatter[row * width + col] != 0. {
                    offer(
                        &mut heap,
                        &mut seen,
                        (stages - 1) * width + row,
                        arrives,
                        start,
                        finish,
                    );
                }
            }
        }
    }

    seed_internal(network, finish, &mut heap, &mut seen);

    // Each static output event may reach another stage after a
    // strictly positive full delay, or leave the cable after a final
    // half delay. The latter is a port waveform breakpoint too.
    while let Some(Reverse(Event { time, node })) = heap.pop() {
        output_times.push(time);
        let stage = node / width;
        let port = node % width;
        let mode = port % modes;
        if port < modes {
            if stage == 0 {
                let out = time + half[mode];
                if out > start && out <= finish {
                    output_times.push(out);
                }
            } else {
                let arrival = 2.0_f64.mul_add(half[mode], time);
                let receiver = (stage - 1) * width;
                for row in 0..width {
                    if scatter[row * width + modes + mode] != 0. {
                        offer(&mut heap, &mut seen, receiver + row, arrival, start, finish);
                    }
                }
            }
        } else if stage + 1 == stages {
            let out = time + half[mode];
            if out > start && out <= finish {
                output_times.push(out);
            }
        } else {
            let arrival = 2.0_f64.mul_add(half[mode], time);
            let receiver = (stage + 1) * width;
            for row in 0..width {
                if scatter[row * width + mode] != 0. {
                    offer(&mut heap, &mut seen, receiver + row, arrival, start, finish);
                }
            }
        }
    }
    output_times.sort_by(f64::total_cmp);
    output_times.dedup_by(|a, b| a.to_bits() == b.to_bits());
    output_times
}
