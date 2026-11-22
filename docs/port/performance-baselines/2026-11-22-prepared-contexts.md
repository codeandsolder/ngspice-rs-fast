# Reused effective device preparation (#53)

Candidate 072d1cd, Acer, campaign nightly, optimized test profile,
managed sccache. Each timing is the median of three alternating
one-shot/prepared samples of 3,000 unchanged-context loads. Both paths
execute identical stamp operations; fresh trial, matrix and RHS on each
load. Each family asserts identical binary64 matrix, RHS and trial state
before timing. This isolates preparation cost, not full sweep throughput.

| Device | One-shot (ms) | Prepared (ms) | Ratio |
|---|---:|---:|---:|
| Contextual resistor | 9.888 | 2.057 | 4.806x |
| BJT | 8.635 | 6.556 | 1.317x |
| MOS1 | 7.417 | 8.509 | 0.872x |
| BSIM4 | 18.616 | 12.674 | 1.469x |

Host heavily contended by concurrent builds. MOS1 result is not a
demonstrated improvement; repeat under lower load. Counted regression
asserts exactly two setters per context transition and zero additional
setters across repeated loads, convergence, observations and linearization.
No numerical tolerances altered.

Reproduce: `cargo +nightly test --locked --test resistor_overrides prepared_context_load_benchmark -- --ignored --nocapture`.

## Reconciled revision (0c624b4)

Repeat after integrating the transient step lifecycle and distortion context
changes, again 3,000 loads and three alternating samples:

| Device | One-shot (ms) | Prepared (ms) | Ratio |
|---|---:|---:|---:|
| Contextual resistor | 15.809 | 13.254 | 1.193x |
| BJT | 12.533 | 4.217 | 2.972x |
| MOS1 | 11.927 | 8.941 | 1.334x |
| BSIM4 | 16.225 | 13.518 | 1.200x |

All stamp/RHS/trial checks bit-identical again. Concurrent build load
remains high. The broad spread between the two runs is a reason not to
interpret individual ratios as precise estimates. Complete reconciled
suite: 943 passed across 20 test binaries, including live C comparisons;
one distinct historical-capture reproduction test excluded because no
matching capture executable is available (see conformance/observations).
Strict production Clippy passed.
