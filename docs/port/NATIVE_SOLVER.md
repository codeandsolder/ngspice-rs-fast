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
  makes pivot sequences, and hence results, bit-identical to C. Each load
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
  #138), which reproduces C's result bit for bit at lower cost.
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
arithmetic, determinant parity and untranslated RHS bits are unchanged.

The [November 22 comparative workload measurements](performance-baselines/2026-11-22-sparse-lifecycle.md)
record unchanged residuals and allocation counts for the recovered lifecycle
candidate versus the accepted scalar baseline.

## Pole-zero trial lifecycle (#27)

The pole-zero search is split into the reusable affine matrix load and
determinant evaluator (`maths::pencil`), scheduler
(`maths::pencil::search`), linked trial arena (`search::arena`) and the
deflated quadratic/Muller kernels (`search::kernels`). A proposal is a
frequency until its determinant is evaluated; an accepted trial receives a
stable arena identity. A collision with a previously accepted root increments
its positive multiplicity without creating a detached trial. The arena owns
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
slot values retain their original meanings. Real-bracket ties, collision
shifts, repeated-root deflation and link order are preserved. The reference
`CKTpzVerify` always accepts a root, so there is no unsuccessful verification
state. The reference never sets `ISANABERRATION`; its `NFlat` counter is only
used in debug printing. Neither is a fabricated production transition.

`conformance/pencil_search_traces.txt` pins five complete traces from C
commit `059066b532f7073fccf4173ef269309eafdc7f0c` on x86_64 Linux using the
SPARSE backend. The ordinary unit test
`analysis::pz::tests::pinned_c_trial_sequences` constructs the same production
pencil loads and compares every binary64 proposal, actual load frequency,
determinant mantissa (including signed zero) and binary exponent. The cases
cover repeated and near-coincident roots, a complex pair, a singular pencil,
and C's iteration-limited notch. The 697 records were captured from the pinned
C binary without modifying its code: function-entry/return breakpoints on
`PZeval` (return: third argument's `PZtrial*`, first two doubles),
`CKTpzLoad` (entry: second argument points to two doubles), and `SMPcDProd`
(return: second argument's complex mantissa and third argument's exponent).
On System V x86_64 those pointers enter via `rdx`, `rsi`, and
`rsi`/`rdx` respectively. Each deck was run with `.control; run; quit; .endc`
before `.end`. Both the pre-refactor Rust trace and the typed search reproduce
all 697 records exactly. A different compiler/libm/architecture should be
investigated if it produces a divergent trace rather than relaxing the
convergence criteria.

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
  `SparseLu::determinant` returns SPARSE's mantissa/exponent pair.
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

- Bit-level fidelity is judged against a `-ffp-contract=off` build of the C
  reference: the reference build fuses multiply-adds, which C's source does
  not ask for, and Rust never contracts implicitly. Against that build the
  conformance decks are bit-identical except those listed in M10.md; against
  the FMA build a few ulp remain. Some decks are decided by that rounding:
  on `multi_analysis_pz` the no-FMA C build and the port both give up on the
  poles after 201 trials while the FMA build finds three, so that golden is
  excluded from `golden verify` (`xtask/src/commands/verify.rs`).
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
