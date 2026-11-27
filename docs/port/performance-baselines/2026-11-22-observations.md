# Bound observations and isolated load workspace

Issue [#54](https://github.com/codeandsolder/ngspice-rs-fast/issues/54).
Baseline `3e0bc82396afb5405fbbd90fc6dc72b8aadf40a8`;
candidate `3e69b4d257110b084f67876607d525ccce19eba6`.
Acer, Ryzen 7 4700U, pinned rustc 1.102.0-nightly (6905905bb);
generic release builds with managed sccache.

## Architecture

* A finalized circuit retains the name/ordinal index already created by its
  topology validation. Topology edits invalidate it and the observation
  plan. An observation binds a topology ID, canonical column name, unit,
  ordinal and typed selector (terminal current, two-terminal power, source
  DC, or device-specific ask). The parameter keyword is retained for the
  effective-device hooks, not reparsed from the column at each point.
  Invalid terminal roles, coincident queried terminals and unresolved
  devices fail at binding; context-dependent parameter support remains
  checked at evaluation. Spectral asks share the binding and reject
  current/power instead of stamping.
* The solver and observations use the same prepared effective-device view,
  including ordered instance replacements. Source DC asks retain the
  supplied sweep value; current asks retain the effective multiplier and
  sign. Column construction reads the bound name and unit.
* One run-owned isolated workspace retains the capacity of an unbound
  sparse triplet matrix, a sparse RHS, per-device trial slots and terminal
  residuals. The RHS sink used by all device stamps selects either the
  ordinary dense solver vector or isolated sorted sparse rows. An isolated
  trial opens just the selected device's state window, still reading the
  same accepted/previous/carry/patch state as a float-mode trial with
  limiting disabled. Trial writes and patches are discarded after each
  stamp. Only workspace capacity persists across points.
* For a requested terminal, one isolated stamp per device per point
  supplies its other asks. Physical rows are resolved once and aliases
  share a residual slot. Traverse matrix triplets in the original
  unfolded contribution order, subtract RHS in terminal order, then
  calculate ground as the negative sum of distinct physical rows in
  first-terminal order. No reassociation, fold or accepted-state mutation.
* Public `Circuit::observations` exposes bound columns; `StampContext::rhs`
  accepts the common RHS sink rather than a raw dense vector.

## Measurements

The same integration test file `tests/observation_bench.rs` was compiled
in each worktree. The direct fixture has 1,024 RC branches and observes
128 resistors plus 128 capacitors at each of 512 changing points. It
requests 512 quantities in total. The full transient has 256 RC branches,
512 observations and a 5 us sine-driven transient with 10 ns maximum
steps. Three interleaved trials of each binary use `perf stat` userspace
task-clock; the median includes fixture creation and test setup. DHAT
reports cumulative process allocations, not memory residency.

| Workload | Baseline task-clock (ms) | Candidate (ms) | Speedup |
|---|---:|---:|---:|
| Direct observation | 150.03 | 40.33 | 3.72x |
| Entire transient | 141.63 | 86.67 | 1.63x |

| Workload | Baseline allocated bytes | Candidate bytes | Baseline blocks | Candidate blocks |
|---|---:|---:|---:|---:|
| Direct fixture only | 12,658,952 | 12,846,601 | 156,552 | 157,745 |
| Direct, 512 points | 2,189,347,177 | 21,215,978 | 576,394 | 162,102 |
| Complete transient | 590,845,366 | 40,805,303 | 466,051 | 55,008 |

The direct workload's fixture-subtracted cumulative allocation drops
from 2,176,688,225 bytes to 8,369,377 bytes (260x), and from 419,842
to 4,357 blocks (96x). Whole-process direct peak was 6,124,752 versus
6,124,753 bytes; transient peak 7,702,739 versus 7,800,226 bytes.
The retained plan's small setup cost replaces repeated per-device
allocations; reduction in cumulative allocated bytes is not a claim of
GB-scale peak memory savings.

All interleaved and DHAT runs matched the baseline values bit-for-bit:
direct checksum `a536b5b508b57645`, full transient checksum
`2cc9468ed55d8b24` (508 timepoints, 771 columns). The new unit tests
exercise ground/cancellation and aliasing, failed binding and stale
topology, interleaved duplicate asks and per-point cache invalidation,
isolated previous-state patches, and recovery from nonfinite results.
Committed C golden, live C asks, swept effective parameters and current
source sign/multiplier tests complement the synthetic benchmark.

## Reproduction

Use the campaign toolchain and managed cache. Build the test binary and
run the cases individually, e.g.:

```sh
cargo +nightly test --locked --release --test observation_bench --no-run
perf stat -e task-clock,cycles,instructions /path/to/test-binary many_saved_devices --ignored --nocapture --test-threads=1
valgrind --tool=dhat --dhat-out-file=profile.json /path/to/test-binary many_saved_devices --ignored --nocapture --test-threads=1
```

Use `fixture_only` for the setup-only allocation baseline and
`many_saved_transient` for the full run; compare the same benchmark copied
into a detached baseline worktree. On ephemeral Cargo infrastructure a
Cargo runner can copy the executable into a temporary benchmark folder
before the managed build removes it.

Logs: `arch-1122-a7c9-54-perf.log` and
`arch-1122-a7c9-54-{baseline,candidate}-{fixture_only,many_saved_devices,many_saved_transient}-dhat.log`
under the campaign logs directory.
