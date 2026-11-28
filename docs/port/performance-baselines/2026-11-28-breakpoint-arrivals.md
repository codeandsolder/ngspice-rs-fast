# Analog breakpoint and exact arrival scheduling (#305)

## Representation

The transient breakpoint table starts with 32 inline scalar slots and
promotes once to the existing two-run ordered index. The indexed insertion
uses a safe mutable B-tree cursor to find the first strictly greater event,
check its predecessor and mutate the same gap. The timestep driver retains
the front value between actual table mutations. No event semantics or
numerical tolerances change. The CPL exact impulse path retains the ordered
arrival map and incident-vector ownership; a single entry search replaces
the separate lookup and insertion, preserving the first two-word arrival
and the arithmetic order of coalesced contributions.

A 32-slot bound was selected from 8/16/32/64-slot sweeps and the actual
source-driven queue occupancy. A cached indexed head was rejected because
maintenance slowed large recurrent queues. A compact ordered CPL queue
and temporary vector pool were rejected because they did not consistently
improve dense/coalescing workloads over the entry-based tree. No new
CPL allocation-capacity accounting or queue representation is required.

## Measurements

Linux x86_64, AMD Ryzen 7 4700U (8 cores), pinned nightly 2026-11-23,
managed-cache release builds from the same claim base
`b7e7b6a296675fa2e9ee84604340bbfb483480fa`.
The original, inline-only and final cursor variants were compared.
The shared host was heavily loaded; thread CPU and hardware counter
measurements, not elapsed time, are the primary evidence.

Exact queue implementation, interleaved thread CPU medians (ms):

| Workload | Original | Inline + indexed clear | Final cursor |
| --- | ---: | ---: | ---: |
| 8 events, 20k recurring cycles | 22.409 | 4.204 | 4.075 |
| 24 events, 20k recurring cycles | 88.333 | 15.474 | 16.992 |
| 32 events, 20k recurring cycles | 121.775 | 118.269 | 77.101 |
| 64 events, 20k recurring cycles | 277.939 | 268.583 | 165.542 |
| 96 events, 20k recurring cycles | 463.871 | 440.010 | 261.565 |
| 65,536 ascending insert/drain | 35.525 | 33.067 | 17.355 |
| 65,536 shuffled insert/drain | 55.558 | 51.503 | 29.562 |

The source-driven benchmark uses the actual current/voltage PULSE accept
functions with controlled accepted timepoints and all source corners, 20
periods, nominal step 0.02 of the period, phased delay and unequal periods.
All three variants match the complete event sequence, including table bits.
Twenty-four cases cover aligned, phased and mixed source kinds, 1–128
sources. Representative phased voltage-source results (thread CPU ms):

| Sources | Peak pending | Original | Final |
| --- | ---: | ---: | ---: |
| 1 | 3 | 0.029107 | 0.013687 |
| 8 | 10 | 0.153978 | 0.054025 |
| 16 | 18 | 0.418428 | 0.130294 |
| 24 | 26 | 0.658523 | 0.222360 |
| 32 | 34 | 0.969497 | 0.711311 |
| 64 | 66 | 2.275001 | 1.636594 |
| 128 | 130 | 5.687292 | 4.264788 |

The 16-/24-event recurring workloads allocate no tree or bucket storage
while inline. DHAT recorded 2,015/4,015 allocations in the original over
1,000 cycles, versus 14/14 shared process allocations with inline storage.
The fixed table grows by 256 bytes per transient driver, not per event.
The 32-event case includes two sentinels, so spills.

The actual prepared CPL material occupancy probe observes exact impulse
queues at each distinct arrival boundary. Peak pending entries for
separable 1/4/16/32-conductor, 16/8/4/4-section cases were
32/64/118/234; for coupled single-stage 4/16-conductor cases 32/512.
The four-conductor multi-stage coupled case selects the unchanged
continuous solver. A stepped expansion matches a whole expansion.
On eight synthetic contractive scattering graphs (2–64 ports) the
single-search entry implementation reduced construction CPU by 3–14%.
On the actual one-conductor 16-section release circuit Callgrind
attributed 12.95M to 10.58M instructions to arrival sends (18.3% lower),
and 26.57M to 24.20M to kernel preparation (8.9% lower).
These are local operation counts, not full-simulator speedups.

## Complete transient execution

Nine randomized interleaved triplets on CPU 3, `perf stat` userspace
cycles/instructions/branches/branch misses, child peak RSS from GNU time.
Each case compares identical binary output before timing. Ratios are
medians of paired runs, not ratios of separately timed wall-clock runs.
The pulse circuits use 20 ns rise, 30 ns fall, 200 ns width, approximately
1 us period, 50 ns maximum step; independent resistor dividers or shared
current loads (1 ohm series supply resistance). The repeated PWL loads
have the equivalent corners. CPL uses matched 50 ohm boundaries,
1 m length, diagonal modal L/C and the specified distributed R
scattering stages.

| Circuit | Inline-only cycles/base | Final cycles/base | Final instructions/base |
| --- | ---: | ---: | ---: |
| Single voltage pulse | 0.982 | 0.968 | 0.992 |
| 16 phased voltage pulses | 1.019 | 0.972 | 0.996 |
| 16 shared-node PWL loads | 0.976 | 0.945 | 0.991 |
| 24 phased shared-node pulse loads | 1.026 | 0.995 | 0.994 |
| 32 aligned shared-node pulse loads | 1.056 | 0.990 | 0.984 |
| 32 phased shared-node pulse loads | 1.061 | 1.024 | 0.997 |
| 16-conductor coupled single stage | 1.006 | 1.008 | 1.000 |
| One-conductor separable 16 stages | 1.004 | 0.989 | 0.997 |

The first inline-only release had workload-dependent cycle regressions.
The cursor and front reuse remove most of them; the 32-source phased
case is still slower on this core despite fewer instructions. Do not
interpret local queue gains as proportional full-simulation gains.

## Reproduction and correctness

Run the focused breakpoints, CPL, companion and source tests with the
repository-pinned toolchain and normal build cache:

```sh
cargo +nightly test --locked --lib devices::breaks::tests
cargo +nightly test --locked --lib devices::cpl::transport::tests
cargo +nightly test --locked --test cpl_lines --test companion_transient --test delay_lines --test transient_sources
cargo +nightly test --locked --lib pulse_schedule_occupancy -- --ignored --nocapture
cargo +nightly test --locked --lib physical_impulse_queue_occupancy -- --ignored --nocapture
cargo +nightly test --locked --release --test breakpoint_scheduler_bench -- --ignored --nocapture
```

The independent sequence oracle uses the direct C successor-first rule
with a vector. It compares all event bits, including signed zeros,
duplicate events and terminal inversion, across 1,124,288 transitions
(256 forced-spill tables, peak pending 279). Repository tests exercise
both initial representations and explicit CPL coalescence, pruning at
capacity, cancellation, first Wide arrival retention and rollback.
The final strict production lint gate and 84 focused tests/probes passed.

Full unfiltered binary outputs match the claim base for 23 circuits,
388,899 accepted points and 20,268,482 binary64 coordinates, including
the two UIC source cases and the 16-/64-source repeated PWL loads.
No time-grid resampling or decimal formatting is used in this comparison.
Independent analytic CPL waveforms agree within 7e-15 V for seven
matched separable/single-stage coupled circuits. Resistor pulse
waveforms agree with the exact source/divider oracle within 6.52e-13 V;
RC cases preserve the baseline finite-step integration error.
For the 15 source decks, both C ngspice 47 and the Rust release were
checked against independent waveform solutions at their own accepted
times. Repeated PWL load voltage and current errors were below 3e-15.
The two solvers can legitimately have different internal timesteps;
their point counts or row indices alone are not a physical error metric.

The 32-source phased shared-node cycle regression reproduced on two
additional cores (0: +1.94%, 2: +2.09%, seven paired trials each).
Both runs had ~0.3% fewer retired instructions. This is a real
microarchitectural tradeoff in the measured workload, not an
algorithmic work-count regression. No workload-dependent special case
is introduced to conceal it. In the eight primary comparisons, peak
RSS was lower or effectively unchanged; the largest fixed table
increase is only 256 bytes.

The 64-source phased shared-node load on core 2 (seven paired trials)
also shows +1.91% cycles with 0.18% fewer instructions. Its source-only
scheduler was 1.39x faster. Retain the general representation for the
substantial scheduler/allocator improvements and the measured whole-path
gains in other circuits, with these full-path tradeoffs recorded.
