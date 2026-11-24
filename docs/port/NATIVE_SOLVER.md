# Native solver: implemented APIs, policies and limits

Current reference for the port's linear algebra and how the analyses drive
it. Since M10 wave 0.5 (PR #137) the port has no external solver crate: the
sparse LU is a Rust port of Kenneth Kundert's SPARSE 1.3, ngspice's own
solver, and every analysis drives it as ngspice's `spsmp.c`/`niiter.c` do.
The decision record and measurements are in
[NATIVE_SOLVER_PROTOTYPE.md](NATIVE_SOLVER_PROTOTYPE.md) and the M10 tracker
([M10.md](M10.md), wave 0.5); the faer/diffsol era it replaced is described
in [DIFFSOL_FAER_RECOMMENDATION.md](DIFFSOL_FAER_RECOMMENDATION.md) and
[SPARSE_RANK_DIAGNOSTICS.md](SPARSE_RANK_DIAGNOSTICS.md) (both historical).
Its implementation guide (`DIFFSOL_FAER_IMPLEMENTATION.md`), the diffsol BDF
backend and its DAE initialization are recoverable from Git history (tag
`v0.4.0`).

## Modules

| Module | Job | C reference |
| --- | --- | --- |
| `maths::sparse13` (crate-private) | SPARSE 1.3: Markowitz ordering with threshold and diagonal pivoting, MNA preorder, refactor with the existing pivot order, transpose solve, determinant; real and complex scalars | `src/maths/sparse/` (`spbuild.c`, `spfactor.c`, `spsolve.c`, `sputils.c`) |
| `maths::linear` | owned real factors (`SparseLu`, `DenseLu`), the `NIiter` factor/solve step (`solve_ni`, `NiStep`), `solve_last_factors`, `PivotTolerances`, C's error policy | `spsmp.c`, `niiter.c` |
| `maths::sparse` | triplet storage (`SparseMatrix`) and bound circuit patterns (`Pattern`, `ElementId`) | `TSTALLOC`/`spGetElement` |
| `maths::complex` | `A + j omega E` assembly (`ComplexMatrix`) and complex factors (`ComplexLu`), pivot reuse across frequencies | `SMPcLUfac`, `SMPcReorder`, `SMPcSolve`, `SMPcaSolve` |
| `maths::pencil` | pole-zero root search on `det(A + s E)` | `cktpzstr.c`, `nipzmeth.c` |
| `maths::integrator` | trapezoidal and Gear companion coefficients, prediction, truncation estimates, accepted step history | `src/maths/ni/` (`nicomcof.c`, `niinteg.c`, `nipred.c`), `cktterr.c` |
| `maths::dense`, `maths::dense_complex` | small dense systems (`.sens`, `.tf`, `.sp` S/Y/Z conversion) | `src/maths/dense/` |
| `maths::equilibration` | opt-in row/column scaling wrappers; no production caller | `sputils.c::spScale` (behavioural reference only) |

SPARSE carries the permissive Berkeley notice (reproduced in `sparse13.rs`
and `NOTICE`), so it may be ported; KLU is LGPL and is not used. The
configuration matches ngspice's `spconfig.h`: `TRANSLATE`, `EXPANDABLE`,
`DIAGONAL_PIVOTING`, `MODIFIED_NODAL`, `TRANSPOSE`, `DETERMINANT`; no
`MODIFIED_MARKOWITZ`, `SCALING` or `CONDITION`.

## How the analyses drive it

- **Element order.** A circuit binds its matrix to a `Pattern` whose
  `ElementId`s are assigned in the order devices first stamp them, which is
  ngspice's `TSTALLOC` order. SPARSE numbers its internal rows from that
  creation order, and Markowitz ties are broken by it, so this order plus
  C's unknown numbering (`Circuit::rebuild_unknowns`: node voltages, then
  each device's internal nodes and branch currents in setup order) is what
  preserves C's pivot ordering policy; arithmetic improvements can change
  numerical ties and rounding. Each load
  replays the pattern's recorded stamp sequence (per device) instead of
  looking elements up; a device with a fixed element list (the MOSFETs,
  `StampContext::stamp_fixed`) adds its values by element once a load has
  verified the list against that sequence, as C stamps through its
  `TSTALLOC` pointers.
- **Newton (`NIiter`).** `solve_ni` performs one load-then-solve step. A
  pattern never ordered is MNA-preordered once (`SMPpreOrder`). With
  `reorder` (`NISHOULDREORDER`) it calls `spOrderAndFactor`, which keeps the
  existing pivots that still pass the thresholds; otherwise `spFactor` with
  the existing order, reporting a zero pivot as `NiStep::ZeroPivot` so the
  caller reloads and reorders, as `NIiter` does after `E_SINGULAR`.
  `LoadGmin`'s diagonal gmin is applied inside the step. Identical inputs
  (values, diagonal gmin, pivot thresholds) reuse the previous factors (PR
  #138), which reproduces the same factors and solutions without refactoring.
- **AC, noise, `.sp`, `.disto`.** `ComplexMatrix` assembles `A + j omega E`
  with explicit `primitives::Complex` conversion; the first frequency orders
  (`NIACSHOULDREORDER`), later ones refactor with the same pivots
  (`SMPcLUfac`). AC never goes through a real solve.
- **`.tf`.** `solve_last_factors` reuses the factors of the operating point's
  last `NIiter` pass, as `tfanal.c` does.
- **`.pz`.** `maths::pencil` ports C's deflated Muller search: each trial
  loads `A + s E`, reorders as `SMPcReorder(1e-30, 0)` and takes the
  determinant. It retains C's 200-iteration limit (reported after 201
  sequence numbers), returning the roots found so far. The reference's
  aberration flag has no producer in the trial evaluator.
- **Transient.** The only backend is the trapezoidal/Gear companion driver
  ([TRANSIENT.md](TRANSIENT.md), `dctran.c`); each timepoint is a Newton
  solve through `solve_ni`. `backend=diffsol` is an explicit `Unsupported`
  error raised by the `.tran` dispatcher (`src/analysis/transient.rs`)
  whatever the deck's options, and `method=bdf` is an explicit error from the
  companion driver: ngspice has no BDF integrator.

## Packed ordered dynamic stamps (#133)

Frequency-dependent small-signal loads keep one reusable term arena per
LinearSystem rather than allocating a vector for each capacitance group.
DynamicGroups holds ordered, nonoverlapping ranges of E triplets and ranges
into contiguous unscaled source terms. Each group still computes its
omega-scaled products in source order and applies the multiplier/divisor to
the accumulated result, matching the original C per-term rounding. Device
stamps that already contain omega use the explicit prescaled flag; the
unscaled and prescaled modes are never treated as interchangeable.

Group range shape, source-coordinate coherence and order are verified once
as the device stamps them. SparseMatrix attaches a lazily minted operator
identity only when grouping is used; clearing, folding or overwriting the
operator invalidates its old groups. All AC, distortion and sensitivity
consumers validate this identity before numeric iteration, returning an
error for stale/foreign group metadata rather than skipping it. The pole-zero
loader traverses the same validated group sequence, but applies the real
s-coefficient rule (including the divisor) and rejects a prescaled AC
coefficient instead of misidentifying it as capacitance. Complete reassembly
retains the capacity of both the group descriptors and the term arena.

The focused tests are tests/dynamic_group_bench.rs and the existing
complex_solver, pole_zero, MESFET/JFET/HFET and reference-conformance
suites. The ignored benchmark contains both the packed construction and a
test-only historical per-group Vec allocation baseline for DHAT comparison.
These paths do not change solver order, pivot selection or C device physics.

## SPARSE frame lifecycle (#34)

A frame has one explicit ordering/factor lifecycle. Original columns are
unlinked; row linking makes an unordered topology. A threshold reorder
that changes a previously ordered topology and then encounters singularity
retains its partial numbering for a subsequent retry. A complete symbolic
state holds an immutable packed traversal; only a factored state holds
valid numeric LU together with that traversal. Nonfinite or resource
failures require a full ordering again, retaining the old traversal only
if its topology remains unchanged. A zero diagonal in fixed-order
refactor preserves the complete symbolic order for reload/retry.

Reload, clear, scalar conversion and any coefficient mutation invalidate
numeric factors; adding an original structural entry requires a new
ordering. Obtaining an existing element handle alone does not invalidate
factors. A second factorization of existing LU or a failed partial LU is
refused until coefficients are cleared or reloaded; editing factor storage
alone does not reconstruct the original assembly. Immutable factor views
borrow the topology and numeric values,
so only a valid factor view can solve, transpose-solve or compute the
determinant. Singular, zero-diagonal and small-pivot diagnostics carry
their respective external coordinates; nonfinite/resource failures
cannot inherit a stale singular coordinate.

Packed runs validate handle bounds, strict row/column traversal order,
diagonal placement and offset representability at construction. Compact
and wide runs have the same numeric kernels; plan validation is not
repeated on numeric refactor. Markowitz arrays are borrowed for a single
ordering operation, then retained for allocation reuse. They are never
cloned into a retained snapshot and are not consulted while eliminating
an accepted pivot prefix. Only an actual new search initializes their
active counts. Topology and complete plans remain shared across snapshots
and scalar conversions; pivot traversal order, fill-in creation order,
determinant parity and untranslated RHS bits remain preserved.

The [November 22 comparative workload measurements](performance-baselines/2026-11-22-sparse-lifecycle.md)
record unchanged residuals and allocation counts for the recovered lifecycle
candidate versus the accepted scalar baseline.

## Measured factor lifecycle (#105)

`SparseLu::factor_storage` and `ComplexLu::factor_storage` expose the valid
snapshot's external and translated orders, original-handle creation count,
stored factor entries and logical numeric-buffer bytes in constant time.
Original creations include structural zero coefficients. Fill entries are
slots first created by elimination (total slots minus original creations).
For pencil-style structural edits, an original created after fill-in increases
the original count; an existing fill handle reused for another purpose still
has its original creation provenance. The ratio is not numerical nonzero
density or allocated capacity, and numeric bytes exclude topology, traversal
plans, workspace and allocation overhead.

`factor_work` reports only the operation that produced the immutable
snapshot, including immediate reload-and-order retries. A compact internal
attempt outcome tracks the actual path and final status. Order calls include
threshold traversals retaining all old pivots; a search is counted only
when the active Markowitz search starts. A fixed-plan refactor request
that falls back to ordering is counted as an order, not a refactor.
Small-pivot warnings are successful factors; zero, singular, nonfinite,
resource and invalid attempts are separate failures. Coefficients rejected
before entering SPARSE or failure constructing a frame are assembly
rejections, not successful factors. A standalone failed operation returns
the existing numerical error rather than a factor snapshot.

`Pattern::factor_telemetry` reads cumulative cache work and current valid
factor storage under the existing cache transaction lock. It includes
real/complex retypes, successful bit-identical factor matches (even if the
subsequent RHS is invalid), and success/failure of RHS solves made through
the real shared cache. An NI zero-pivot result is a failed factor attempt;
its later caller-directed reorder is counted separately rather than inferred
to be a retry of identical coefficients. Immediate retries inside a factor
operation are counted explicitly. Complex LU snapshots and their independent
normal/transposed solves do not mutate cache telemetry, nor do standalone
factors constructed from a bound matrix. Their successful solves are measured
by the calling workload. This preserves snapshot isolation, including
concurrent solves and solutions of old factors after a later factorization.

No numeric inner-loop counters or new locks were added. The shared counters
saturate and remain consistent with the cache; cloning or extending a
pattern creates a fresh record, and discarding a poisoned cache also resets
work and storage. The latest storage is absent when no valid numeric LU is
retained. The counters do not include assembly validation outside the cache
(e.g. construction of a complex operator), detached snapshots, or earlier
pattern epochs. See [PERFORMANCE.md](PERFORMANCE.md) for benchmark output
and interpretation.

## Pole-zero trial lifecycle (#27)

The pole-zero search is split into the reusable affine matrix load and
determinant evaluator (`maths::pencil`), scheduler
(`maths::pencil::search`), linked trial arena (`search::arena`) and the
deflated quadratic/Muller kernels (`search::kernels`). A proposal is a
frequency until its determinant is evaluated; an accepted trial receives a
stable arena identity. A collision with a previously accepted root increments
its positive multiplicity only if the deflated determinant still has a zero
there, without creating a detached trial. The arena owns
both links, immutable linked frequencies and the anchor. Insertions check
finite frequencies and live link targets; pruning retires an ID permanently.
C’s sample-collision scan can stop before visiting a lower-frequency
complex trial, so link order is not a strict frequency-order invariant.
The arena preserves the scanned predecessor, including this collision path,
until root verification prunes nearby and complex samples. Invalid or retired
trial identities produce numerical errors.

Neighborhoods distinguish missing-middle, missing-right and missing-left
initialization (in that priority), a complete real bracket, incomplete complex
seeds, the complete Muller triple, and a center minimum. A strategy is either
navigation along the linked list or an evaluation with its required IDs.
Bracket regions and movement directions are explicit. C's selection predicate
uses the center minimum before the imaginary part of the left slot; the other
slot values retain their original meanings. Real-bracket ties, collision shifts and link order preserve the reference
structure. Acceptance instead requires a local deflated-determinant check;
conjugate pairs contribute their quadratic polynomial, not a squared
magnitude at complex frequencies. The reactive row/column support bounds
the maximum degree after drive-column replacement. A distinct-frequency
probe distinguishes an exact first root from an identically singular pencil. The reference never sets `ISANABERRATION`; its `NFlat` counter is only
used in debug printing. Neither is a fabricated production transition.

`conformance/pencil_search_traces.txt` pins five complete traces from C
commit `059066b532f7073fccf4173ef269309eafdc7f0c` on x86_64 Linux using the
SPARSE backend. The ignored diagnostic
`analysis::pz::tests::report_pinned_c_trial_sequences` constructs the same
production pencil loads and reports the first divergence in the proposals,
load frequencies and determinant representation. The cases
cover repeated and near-coincident roots, a complex pair, a singular pencil,
and C's iteration-limited notch. The 697 records were captured from the pinned
C binary without modifying its code: function-entry/return breakpoints on
`PZeval` (return: third argument's `PZtrial*`, first two doubles),
`CKTpzLoad` (entry: second argument points to two doubles), and `SMPcDProd`
(return: second argument's complex mantissa and third argument's exponent).
On System V x86_64 those pointers enter via `rdx`, `rsi`, and
`rsi`/`rdx` respectively. Each deck was run with `.control; run; quit; .endc`
before `.end`. The pre-refactor implementation reproduced those records exactly; the
accuracy-oriented implementation intentionally need not do so. Ordinary
tests instead check independent characteristic-polynomial residuals,
complete root sets and multiplicities, and the live C cases that converge.

## Error policy (C's)

- `spSINGULAR` and `spZERO_DIAG` are errors (`E_SINGULAR`). Circuit solves
  name the unknown as `NIiter`'s warning does (`singular matrix:  check node
  X`); a zero pivot met while refactoring is first retried with a reorder.
- `spSMALL_PIVOT` is accepted, as in C. C is silent; the port prints a
  warning (each distinct message once, at most six per process, like
  `NIiter`'s `msgcount`). Warnings never change a result.
- An unknown that no element references (a node joined only to current
  sources) has no SPARSE row; as in C, circuit solves leave the right-hand
  side in its solution entry, with a warning. Standalone factorizations
  (`SparseLu::new`, `DenseLu`) reject such a system instead.
- There is **no** condition estimate, rank probe, residual check or
  equilibration on circuit solves, because ngspice has none: each would
  reject or alter circuits C simulates. Non-finite coefficients, duplicate
  summation overflow, non-finite right-hand sides and non-finite solutions
  are still errors, and dimensions are validated before the solver runs.
- `.options pivtol` / `pivrel` (`CKTpivotAbsTol`/`CKTpivotRelTol`, defaults
  `1e-13` / `1e-3`) reach every reorder through `AnalysisContext::pivot`.
  Values C's `spOrderAndFactor` would silently replace (`pivtol < 0`,
  `pivrel` outside `(0, 1]`, non-finite) are rejected.

`tests/solver_robustness.rs` checks degenerate decks (ideal-source loops,
floating nodes, current-source cutsets, near-singular, badly scaled, zero
R/C/L) against C's answer or C's class of error.

## Opt-in original-operator residual telemetry (#104)

Ordinary C-compatible DC/Newton/AC solves still perform no implicit
equilibration, residual acceptance or conditioning-based rejection. The
maths-library diagnostic, `OriginalResidual::<f64>::real` or
`OriginalResidual::<Complex>::complex`, instead borrows the **unscaled**
original assembled entries. The constructor indexes normal or transposed
orientation once; `evaluate` and `evaluate_many` reuse that
index for one or several original-unit right-hand sides. Disabled telemetry
has no runtime overhead and no duplicate numeric operator storage.

`ComponentwiseBackwardError` is a non-rejecting, dimensionless
per-row measure. For real rows it is
`|Σ a_rc x_c - b_r| / (Σ |a_rc| |x_c| + |b_r|)`.
Complex residuals evaluate the real and imaginary products separately and
take the larger normalized component; the denominator bounds both using all
four absolute real/imaginary cross-products for each coefficient/solution
pair, plus both RHS components. An empty zero row with zero RHS has error
zero. The metric does not imply a forward-error/condition-number guarantee.

Exact binary64 product accumulation reuses `primitives::ExactSum`
to avoid cancellation loss, overflow and underflow in the numerator and
denominator, including products individually beyond binary64 range. They
retain separate power-of-two exponents until the final quotient. Reports
contain per-row ratios and their maximum. Invalid dimensions, non-finite
operators, RHS or solutions are errors; poor scaling or a large finite
backward error is *reported*, not rejected as singular.

The diagnostic is opt-in, with `O(nnz)` integer traversal indices
and `O(n)` reported errors, no deep copy of the numeric operator
and no per-RHS reindexing. Ordinary LU/C parity are unchanged.

The optional `EquilibratedSparseLu::solve_refined` now uses the
same exact original-unit quality and once-rounded correction RHSs. It takes
an explicit `RefinementPolicy { max_steps, target_backward_error }`
(maximum 8 rounds) and returns `RefinedSolution` with its best
finite solution, final componentwise quality, count of accepted corrections,
and a structured status: target met, rounds exhausted, no improvement,
unrepresentable correction, or a correction-solve failure message. Every
accepted correction must strictly improve the maximum original-unit error.
The refined path may return a nonconverged physical solution with its
quality/status for a caller to decide; it does not silently relax the target
or reject a solution on a global-norm conditioning heuristic. This is an
opt-in numerical experiment, not an automatic replacement for C's solver.

Regression tests cover maximum-scale and minimum-subnormal real products,
exact cancellation, complex phases and unrepresentable imaginary products,
normal/transposed and multiple-RHS comparisons and invalid dimensions and
finite inputs.

## Library API contracts

Running a deck's analyses from Rust, as the CLI does (`RunConfig` applies the
deck's `.option` cards):

```rust
use ngspice_rs::analysis::RunConfig;

let config = RunConfig::from_netlist(&netlist)?;
let mut circuit = config.circuit(&netlist)?;
let plan = config.plan_for(&netlist.analyses[0])?;
let plot = plan.run(&mut circuit, &config.context())?;
```


The resolver evaluates deck expressions, validates analysis syntax and options,
and builds the typed analysis-specific payload once. Batch scheduling consumes
`config.plans_for(&netlist.analyses)?`, the same plans subsequently executed.
Circuit-dependent identities bind against the circuit when running a plan.

- `SparseLu::new(&matrix, reuse)` returns owned factors; the matrix stays
  plain, cloneable stamping storage, and mutating it never changes existing
  factors. `reuse = Some(previous.symbolic())` refactors with that pivot order
  only for an identical assembled pattern and errors otherwise; exact
  duplicate cancellation can change the pattern.
- `SparseLu::solve` and `DenseLu::solve` accept repeated right-hand sides;
  `SparseLu::determinant` returns a signed binary mantissa and an i64 binary exponent,
  with magnitude in [1, 2) for nonzero real determinants.
- `SparseMatrix::solve` is a convenience that factors the current assembly.
  Empty or non-square systems are errors, not empty solves.
- `ComplexMatrix::{factorize, factorize_reusing}` and `ComplexLu::{solve,
  solve_transposed, determinant}` mirror the real API; `solve_transposed`
  does not conjugate (`SMPcaSolve`).

Unit tests live beside each module; `tests/linear_solvers.rs`,
`tests/complex_solver.rs` and `tests/solver_robustness.rs` exercise the
public interfaces (pivoting, repeated right-hand sides, mutation after
factorization, pattern changes, dimension/empty/non-finite/overflow errors,
singular systems).

## Fidelity and limits

- The solver preserves the SPARSE policy, not intermediate C rounding.
  The factor frame stores both triangular factors with their shared raw
  diagonal, plus a numerical cache of normal reciprocal pivots. Ordinary
  updates use the reciprocal-based fast path; cancellation-sensitive
  updates recover from the original cross-product before dividing.
  When the reciprocal or normalized multiplier is outside binary64
  range (including subnormal precision), use the raw pivot and
  conditioned cross-product instead. A nonrepresentable multiplier
  alone does not invalidate the factors. A nonrepresentable stored
  Schur coefficient does; an overflowing final solution is an RHS
  error and retains the valid factors. Packed row and column runs support
  either solve orientation. The forward pass retains raw numerators;
  complex backward substitution also uses raw scatter. Real backward
  substitution normally gathers normalized terms in row order. A local
  roundoff bound rejects unresolved cancellation, and normalization or
  product range failures also reject that route. Rejected RHS values
  are replayed using raw scatter from the original RHS, independently
  for each SIMD lane. The guard is not a global conditioning certificate.
  See [BSIM_NUMERICAL_DIAGNOSIS.md](BSIM_NUMERICAL_DIAGNOSIS.md) for
  the independent C replay motivating the change.
  Exact product accumulation distinguishes
  proportional rows from genuine tiny nonzero determinants. Determinants
  multiply raw pivots directly, not their rounded inverse reconstructions.
  Determinants are accumulated with binary normalization, with no decimal
  conversion in the pole-zero path. The historical no-FMA C failure on
  `multi_analysis_pz` is no longer reproduced: the notch and multi-analysis
  cubics are validated against independently derived polynomial residuals
  and Vieta coefficients. The historical C traces remain optional diagnostics.
  Existing reference tolerances are unchanged.
  The complete #91 golden audit verifies 219 of 220 fixtures (all eleven
  pole-zero fixtures). The exception, `sens_jfet2`, remains a sensitivity
  finite-difference accuracy issue, not a pole-zero or factorization
  failure. For example, its physical DELTA derivative independently
  evaluates to approximately 0.00116748671553, whereas the historical
  C 1-ppm finite difference gives 0.00116747694002 and the #91 finite
  difference gives 0.00116749333592. Both miss the existing precision
  bound on the physical derivative; the latter is closer. Keep the
  historical capture and repair the derivative/reference calculation
  in [#144](https://github.com/codeandsolder/ngspice-rs-fast/issues/144).
- libm functions (`exp`, `log`, `pow`) differ between platforms, so
  bit-exact assertions on results that depend on them hold only on the
  oracle's platform (macOS arm64); elsewhere the tests use tolerances.
- Markowitz ordering fills large 2D meshes far more than faer's COLAMD
  ordering did: on the 50k-node mesh the port needs 1.8x C's instructions
  where the faer control needed 0.32x (the "After wave 0.5" benchmark in
  M10.md). SPARSE stays for fidelity; an alternative ordering is the lever if
  large meshes matter.
- The M10 wave 0.5 performance items are done: device-load cost (0.5e:
  MOS inverter transients 2.4–2.6x C wall time), complex AC assembly (0.5f)
  and plot memory (0.5g). `.ac` plots are not streamed yet (M10.md, 0.5g).

## Validation

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
cargo +1.89.0 test --workspace --locked
cargo xtask golden verify
NGSPICE_BIN=/absolute/path/to/ngspice cargo test -p ngspice-rs --test c_reference --locked -- --ignored
```

The no-FMA oracle and trace tooling used for bit-level comparison live
outside the repository; see [VERIFICATION.md](VERIFICATION.md) for the
committed C-golden workflow.

### Aggregate elaboration resources

The device-generation limits apply to the materialized circuit, not just the
flattened input cards. `ElaborationLimits` defaults to 250,000 device records,
1,000,000 distinct non-ground nodes including setup-private nodes, and
4,000,000 declared scalar device-state slots (32 MiB per complete f64 state
vector). Call `Circuit::with_elaboration_limits` for incremental construction,
or `Circuit::from_netlist_with_limits` for a complete deck with explicit
mutual-inductor and elaboration limits. The complete-deck path also uses the
device limit for initial subcircuit expansion.

The planner charges existing records and one minimum record per incoming
card, then binds exact URC and TXL expansion counts and distinct written
terminals before constructing any ladder. It retains plans only for those
special cards; ordinary cards remain borrowed. Ordinary private nodes and
state counts are checked when the factory binds them, and every planned
expansion is checked against the actual remaining budget and verified against
its resulting counts. Checked sums, node identity range, state allocation
layout and fallible staging capacity reject overflow. The final device
vector reserve takes place inside the node transaction: allocation failure
or a later card error leaves the previous nodes, devices, origins and
finalized analysis bindings unchanged. Manual node/device additions and
replacement obey the same aggregate limits. This bounds materialized records
and state slots, not arbitrary strings, model caches or dynamic analysis
history storage. Standard allocator-backed boxed devices and map/string
construction remain subject to the system allocator.

### Independent right-hand sides

Real and complex sparse LU expose a reusable typed `SolveBatchWorkspace`.
RHS and solution buffers are RHS-major in external node order; four
independent RHS vectors share one factor traversal, using separate portable
SIMD real/imaginary planes. Partial tiles use the scalar substitution
kernel. Workspace storage is bounded by one full tile plus one scalar
tail. Pivoting, normal/transpose ordering, untouched external rows and
scalar exponent/cancellation recovery are unchanged. Nonfinite input or
output reports the failing RHS index.

RF excitations and RF noise adjoints are prepared and solved in port-order
tiles; the numeric cell budget accounts for the tiles and optional scalar
tail. DC and AC sensitivity prepare parameter residuals in the existing
device-record walk, solve independently against the retained factors, and
emit in parameter order. Ordinary single-excitation noise solves stay
scalar. No new factorization or reassociation is introduced.

The opt-in sparse RHS benchmark exercises real and complex normal and
transpose solves at 1, 2, 4, 8, 32 and 64 RHS with warm reusable buffers:

```sh
cargo +nightly test --release --lib sparse_rhs_batch_benchmark -- --ignored --nocapture
```

### Guarded normalized real backward substitution (#154, #193)

Real SPARSE follows C's normalized backward **row gather** where its
coefficients and operations are representable. Every RHS retains the raw
numerator until a complete guarded solve succeeds; unresolved cancellation,
non-normal multipliers, range failures or non-finite values trigger a raw
scatter replay of that RHS (independently for SIMD lanes). The guard cannot
certify global condition number or justify changing physical model outputs.

For each row, let `m[0]` be the magnitude of its initial normalized
numerator and `m[j]` the cumulative nonnegative magnitude after subtracting
term `j`. The original per-prefix local error estimate was
`2*eps*m[0] + 4*eps*sum(m[1..n])`. Every prefix is bounded by the
final accumulated magnitude, so the conservative
`(4*n + 6)*eps*(1 + 2*n*eps)*m[n]` also bounds the rounding of
positive prefix-error sums for every representable row length (through
`u32::MAX`). The term count comes from the immutable packed row
slice; zero numerical coefficients may overestimate it safely.
`RealGatherRow` now computes this factor once per row, rather than
multiplying and accumulating a separate error estimate for every term.
The exact normalized coefficient, subtraction order and per-term
normality masks are unchanged for scalar and four-lane solves.
Unrepresentably large row lengths take the existing raw path instead
of truncating their error count. Cancellation checks are deliberately
conservative, not an attempt to force a C source-current bit pattern.

The focused regression compares the new validity predicate to #154's
original per-prefix predicate across 2,048 deterministic variable-length
rows: every accepted new row must be accepted by the original, and the
pre-fallback normalized numerical result must match bit-for-bit. The four
strict BSIM4 #154 external-source C oracles (1e-9 relative), nine BSIM4
device cases, sparse pivot/range/transposed/heterogeneous batch cases and
committed goldens remain the required acceptance constraints.

Reproduce retained-factor solve performance separately from factorization:

```sh
cargo +nightly test --release --locked --test solver_workloads_bench \
  real_backward_substitution_crossover -- --ignored --exact --nocapture
```

This opt-in test uses a 512-node tridiagonal ladder and a 400-node mesh,
checks manufactured solutions and 1/4/32-RHS batch equivalence for normal
and transposed solves, then reports medians of three 16-iteration batches.
Benchmark interpretation and paired historical data are maintained in
[the #193 working log](https://app.notion.com/p/422f74be902081c38049d95a05e0f3f7),
not in this current architecture reference.

### Symmetric Jacobi validation of coupled inductances (#148)

The dense fallback checks the signed spectrum of a normalized mutual
inductance matrix only after inexpensive Gershgorin certificates fail. The
matrix is symmetric by construction. Each cyclic Jacobi step now rotates an
off-pivot pair once and copies the rounded result to both mirrored entries,
while explicitly annihilating the pivot coupling and updating the two
diagonals via its stable bounded tangent. Independent row and column rotations
were producing false skew, preventing convergence for otherwise valid dense
coupling (the exact 64×64 Hadamard example in `tests/mutual_jacobi.rs`).

Stop only when the omitted symmetric off-diagonal Frobenius norm is at most
`4*sqrt(n)*epsilon*||A||_F`. This is a backward perturbation: Weyl's bound
then limits each eigenvalue change to that norm, in turn at most
`4*n*epsilon*||A||_2`. The mutual validation acceptance budget is
`64*n*epsilon*||A||_2`, so the solver residual is at least sixteenfold
smaller than the physical decision margin. The 100-sweep numerical work
budget, power-of-two scaling at extreme magnitudes, input symmetry checks
and explicit failure behavior remain. This does not turn the eigensolver
into an approximate general-purpose public API.
