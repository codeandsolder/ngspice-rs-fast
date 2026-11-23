# Behavioural symbolic compilation

The value tree remains unsimplified, and its instructions are interned in
postorder by exact operation, operand order, and floating-point bits. C
`PTdifferentiate` rules are applied symbolically, not replaced by automatic
differentiation or algebraic reassociation.

Each source instruction has one default symbolic partial for a quantity
absent from its differentiable dependencies and a sorted range of overrides
for specific quantities. A default is computed using the actual C `mkb`
and `mkf` folding (constant pairs first, then zero and unit identities).
It is not invariably zero: constant division or an invalid function
slope may form a non-finite constant, and signed zeros must be preserved.
Child override ranges are merged once per source instruction; one arena
stores their entries. Structurally identical derivative operations share
slots. A ternary condition is evaluated as part of the value, but its
condition-only dependencies are not differentiated.

The compiler charges each default/override differentiation against a
131,072-rule application budget. The source AST retains its size/depth
limits. The interner caps all instructions, including intermediate
operations discarded by simplification, at 262,144. Thus dense dependency
sets cannot cause unbounded intermediate allocation. Sparse source trees
are no longer rejected on a source-node-count times quantity-count estimate.

After differentiation, an iterative reachability traversal emits only
instructions reachable from the value followed by each derivative root.
Operand order matches the old tree-lowering postorder; even an invalid
non-finite folded constant must not be evaluated if the enclosing operation
simplified it away. The runtime still evaluates selected ternary branches
lazily. PWL/TABLE lists retain their existing shared immutable storage.

The previous tree compiler lives only in tests as an independent reference.
Unit tests compare the entire reachable indexed plan, value/gradient bits,
and diagnostics for constant, sparse, dense, piecewise, powers, PWL and
TABLE expressions. Run the opt-in compiler benchmark:

```sh
cargo +nightly test --locked --lib sparse_and_dense_compilation_benchmarks -- --ignored --nocapture
```

It measures median wall time and counts derivative-tree Arc allocations
from the reference (zero by construction in the new indexed compiler),
plus retained plan slots. The Arc count is not a whole-process allocation
count.
