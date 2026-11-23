# Selected device observations: scratch lifecycle

A bound observation plan records the physical terminal rows that each device
may need. It is independent of numerical state. The observation workspace is
owned by the prepared analysis run, never by a shared device, and is dropped
when the run ends.

## Sparse current storage

A loaded-terminal record holds a validity epoch and an initially empty current
buffer. The buffer is allocated on its first selected current/power request,
sized to distinct physical rows plus the ground residual slot. It is reused
for every later point. An objective selecting one current among 1,000
registered loads initializes one buffer, not all 1,000. Row layouts and
metadata remain in the bound plan, without rediscovery per sample.

The begin operation advances one epoch instead of visiting all loads.
A selected device is stamped only when the load's recorded epoch differs
from the current epoch. The record is updated only *after* the isolated stamp
and residual calculation both succeed; failures cannot mark partial data valid.

Generation zero is reserved for never-valid entries. When the 64-bit counter
rolls over, all metadata epochs are cleared once before epoch 1 is reused.
Previously allocated scratch capacity remains intact. This is a one-in-2^64
maintenance operation rather than an O(number of bound loads) per-point scan.

Each nonempty point evaluation starts a new epoch, isolating rejected
transient trials, accepted points, source replacements and independent
candidates. The physical load still preserves row and ground aliasing,
accumulation order and selected current/power identity.

## Complexity and scope

For N registered loads, O(N) compact per-load metadata is eager. Current
scratch bytes are proportional to the terminal widths of *actually requested*
loads rather than all bound loads. Once selection is warm, this scratch
does not allocate per point. Returned output vectors and the numerical
solver have their own allocation policies.

Preparation and terminal binding are unchanged. Full-output requests do not
require priming; each requested load initializes its scratch when first used.

## Verification

Focused unit tests cover interleaved current and power selectors, terminal
aliases, isolated state history, cold buffers, epoch invalidation, exact wrap
and scratch capacity reuse. The prepared-run workload measures 1-of-24 and
1-of-1,000 selected loads for 1,001 points, then executes full selection.
The 1,000-load measurement is excluded from ordinary CI and can be selected
explicitly in a quiet benchmarking environment. Benchmark the already
prepared circuit, not repeated topology construction.

