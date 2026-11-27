# Topology-bound independent source coefficients

Issue #163 follows source binding #141, #120 and #133. Independent-source
linear assembly reuses immutable recipe metadata but never caches numerical
row indices in a source device or global table.

## Representation and lifetime

- LinearSource has an inline SourceRows value containing zero to two ordered
  (row, coefficient) pairs. A voltage source contributes its branch equation;
  a current source contributes negative positive-terminal and positive
  negative-terminal KCL coefficients, omitting ground. Coincident rows remain
  distinct and in the original order. A third row is a structured error,
  never silent truncation.
- Source identity is an Arc<String> shared with the IndependentSource. Its
  immutable waveform is Arc<Waveform>, shared across independent AC assemblies.
  Cloning these handles never copies the source string or PWL/knot list per
  frequency. The RF-PORT accumulated waveform has its own Arc; it cannot alias
  an unrelated predecessor. Transient analysis timing resolves a fresh
  analysis-local waveform, not a mutation of the original.
- Rows are resolved at each actual LinearContext binding using finalized
  MnaUnknowns and branch indexing. No address-keyed mutable cache exists;
  re-finalizing another circuit or changing a topology cannot reuse stale
  row indices. An A/B/A temperature or source-parameter context retains the
  immutable source identity but not previous system state.
- Both ordinary and biased small-signal paths pre-reserve one sources vector
  from the finalized count of independent V/I devices, then append in the
  original load order. Capacity failures become typed circuit errors.
- DC, AC, transient, noise, distortion and RF clients consume the same inline
  rows. NoiseSolve stores its bound input rows inline and checks their equality
  on reassembly.

## Allocation measurements

Existing #141 benchmark is the ignored/manual
tests/behavioural_workspace_bench.rs,
measure_prepared_behavioural_stamp_workload. It executes 3 x 1000 prepared
Newton loads and 1000 separate AC Jacobian assemblies using five nonlinear
B sources and three independent voltage sources. Valgrind DHAT:

| Metric | Prior #141 accepted | After #163 | Difference |
|---|---:|---:|---:|
| Whole-process allocations | 13,814 | 7,818 | -5,996 (-43.4%) |
| Whole-process allocated bytes | 2,583,383 | 2,079,411 | -503,972 (-19.5%) |
| Per-source row Vec | 3,000 / 48,000 bytes | 0 | eliminated |
| Per-source name clones | 3,000 / 7,000 bytes | 0 | eliminated |
| Source collection Vec | 1,000 / 800,000 bytes | 1,000 / 352,000 bytes | 56% smaller |

Previous snapshot:
STATE/logs/worker-arch-1122-b6d9-issue141/dhat-new.json.
Current snapshot:
STATE/build/arch-gpt6-1123-0951-a9f/issue163-dhat.json.

Both runs execute the same test. The older #141 result predates other integrated
changes, so unrelated differences can affect whole-process totals. The row
and name sites were individually attributed by the original DHAT call stacks
and are absent in the new implementation. Newly constructed LinearSystem
objects still allocate their source collection once; this is not a claim of
zero overall allocations or no matrix reconstruction. Valgrind times are not
comparable native performance benchmarks.

To reproduce after compiling the manual benchmark with the pinned nightly:

    valgrind --tool=dhat --dhat-out-file=profile.json \
      /path/to/behavioural_workspace_bench-BINARY \
      measure_prepared_behavioural_stamp_workload --ignored \
      --nocapture --test-threads=1

## Numerical and lifecycle verification

tests/source_rows_allocation.rs covers ground omission, signed DC/AC RHS,
duplicate terminal rows, branch unknowns, rejected overfill, source ordering,
DC sweep override, shared Arc identity across repeated A/B/A contexts,
independent layouts with different node numbering and analysis-local transient
binding. An ignored 1000-assembly workload reports shared handles without an
unsafe global allocator.

Existing linear_analysis, noise_analysis, sources and source_waveforms
integration tests cover exact source signs, waveforms, transient events,
RF and solver cases. The serial campaign validator runs strict lints and
the C-derived golden oracle corpus. No physics or numerical tolerances change.
