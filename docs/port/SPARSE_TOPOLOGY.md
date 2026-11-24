# SPARSE topology exchange invariants

Issue #165 unifies the SPARSE 1.3 row/column exchange logic in the linked
topology without changing original sparse-element handles, numeric storage,
pivot selection, external numbering, or the public solver API.

## Structure and orientation

Each element has a stable index in the topology's elements array and
two singly linked successors. Row lists are strictly ordered by column;
column lists are strictly ordered by row. Every element occurs exactly
once in each list. The first_in_* headers select the first entry or NONE.

LinkedAxis selects the direction using private, zero-sized AlongRows and
AlongColumns types. Each orientation supplies head/after link access, its
coordinate and successor getters/setters, and a permutation-map swap.

The associated Orthogonal axis enforces which list must be relinked.
An exchange_axis of AlongRows walks and swaps row lists while updating
column membership through exchange_list_elements of AlongColumns; the
column case is the type-level inverse. Both compile to monomorphized
functions without trait objects, heap workspaces or runtime dispatch.

The shared relinker retains SPARSE's four exact cases: missing first,
missing second, adjacent elements and nonadjacent elements, in their
original predecessor rewiring order. Duplicate coordinates are never
inserted. A same-index permutation is an explicit no-op. Internal/external
permutation maps update only after the relink walk is complete.

Sparse::exchange_rows_and_cols still owns Markowitz counts, parity toggles
and diagonal relinking. Identically numbered row/column exchanges retain
their original even permutation parity. Existing copy-on-write behaviour
when a topology Arc is shared is unchanged; the relinking itself needs no
allocation and the element handles remain stable.

## Independent verification

The in-module deterministic structural oracle directly exercises Topology
and separately tracks every original handle's expected coordinate and
internal/external permutations. After every operation it verifies both
list visit counts, strict coordinate ordering, inverse maps, no dangling
links and no cycles. It covers multiple sizes and densities, self-swaps,
first/last swaps, repeated shuffles, and single-element/empty-list cases.

The external tests/sparse_topology_exchange.rs checks the public real and
complex factor/solve paths, including transpose, shifted permutation
matrices, duplicate input triplets and singular matrices. Expected right
hand sides are computed independently from dense matrix products.

Existing SPARSE, complex, solve/refactor and C-derived tests remain the
authority on pivot selection, floating-point fidelity, ordering parity
and numeric storage. No tolerance was loosened.

## Repeatable performance workload

An ignored 128x128, 100-factorization test exercises repeated full numeric
factorization. Run on the campaign-pinned nightly with managed sccache:

    cargo +nightly test --locked --test sparse_topology_exchange \
      benchmark_repeat_permuted_factorization -- --ignored --nocapture

If replacing the linked four-case algorithm with an unlink/reinsert
alternative later, benchmark identical input/compiler flags on the
original before adopting it. This refactor preserves the original case
sequence and makes no throughput increase claim.


## Exact accepted-base numerical comparison

The new direct dense-matrix test initially had uncalibrated forward-error
limits 2e-10 (real) and 3e-10 (complex). A 23x23 complex transpose case and
one 23x23 real transpose case exceeded these although the structural
permutation oracle passed. To investigate without altering simulator math,
the entire new test was run on an independent read-only git archive of the
**exact accepted base** a0b63a291bbb02ef1cc75f952b12d970f9b69a25,
with the same pinned nightly and managed sccache. Both builds produced
**identical** reported forward errors for every permutation/size and both
solve directions. The largest was 4.57379013009883613e-10 (23x23 complex
transpose, shift=2); largest real transpose error was
3.35920402605438539e-10. The 13x13 complex transpose result
Complex(re=0.8499999999947105, im=-4.1395642353762397e-10) was
**bit-for-bit identical** between accepted and refactored algorithms.

The new public-solve test therefore bounds forward error by 1e-9, about
twice the measured accepted-base maximum, **not** a claim that all double
precision SPARSE solves reach machine epsilon. The exact structural oracle
separately requires zero coordinate, handle or permutation differences
after every swap. Both accepted-base and new diagnostic traces are under
the worker build folder: issue165-full-diagnostics-base.log and
issue165-full-diagnostics-current.log. This is one-time integration
evidence, not a persistent numerical-lock-in API or an accuracy ceiling.
