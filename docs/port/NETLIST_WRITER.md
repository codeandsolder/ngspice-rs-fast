# Bounded normalized netlist writing

The semantic writer emits cards in source order, preserving normalized
spelling, duplicate setters, include directives, and subcircuit boundaries.
It never flattens resolved include contents. Its compatibility target is
semantic equality after reparsing with the same parser options, followed
by the byte-identical normalized fixed point.

## API and bounds

`write_netlist_to(&netlist, &mut sink, limits)` is the emission kernel.
It accepts any `std::io::Write` and does not flush it.
`write_netlist` collects the same emission into an owned UTF-8 string.
There is no separate collector-specific serializer.

`WriteLimits` defaults:

| Budget | Default | Charged for |
| --- | ---: | --- |
| bytes | 64 MiB | All emitted UTF-8 bytes including line breaks and indent |
| card_bytes | 16 MiB | Each complete emitted line, including the title |
| cards | 1,000,000 | Ordered card occurrences in each visited scope, including skipped include occurrences |
| depth | 64 | Nested subcircuit scopes; cannot exceed the parser limit |

The reusable card buffer grows geometrically using fallible reservation.
It is bounded by both the card budget and the remaining total-byte budget.
Validation and formatting happen before the current card is emitted.
Raw output cards are already borrowed strings and require no reconstructed
buffer. A scope skipped because of its include provenance is not recursively
traversed. These are emission bounds, not a process-wide heap quota:
the editable input AST already exists, and validation still uses the
parser's own bounded expression trees and nonlinear-form checks.

An error may leave earlier complete cards in the sink; an I/O error may
also leave part of the current card. Callers requiring atomic file
publication must stage the result and rename only on success.
An I/O error carries the source card's path. The writer does not close
or flush the caller's sink.

## Borrowed formatting and validation

Names, setter values and expression spellings are borrowed. Scalar and
vector setters write into the same buffer directly, including PWL,
initial-condition vectors, and nonlinear source forms. The tokenizer
and writer's scalar checks share the same lexical scanner. Ordinary
words, numeric tokens, and unescaped quoted text do not allocate an
owned token or logical source line during scalar validation.

Public ASTs are editable, so skipping semantic validation based only
on stored spellings is unsound. Individual parameter and behavioural
expressions are reparsed and compared using borrowed semantic comparison.
For .param and .func cards, a single complete-card reparse verifies both
the grammar boundary and the stored expression trees. Node hints no
longer synthesize an owned parameter assignment or clone expression trees.
Option expression values and node-hint literals also verify their
stored semantic values.

## Validation and benchmark procedure

The deck-writer suite includes all conformance fixture round trips,
exact byte/card/depth boundaries, oversized malformed input, and sink
failures at every byte position of a representative nested deck.
The behaviour grammar and tokenizer suites check shared lexical and
nonlinear semantics.

The ignored tests in `tests/netlist_writer_bench.rs` exercise 128 model
lists with 256 setters each, 128 PWL sources with 512 points each, and
32 nested scopes with 2,048 devices and parameter/behavioural expressions.
Each workload checks semantic equality and byte-identical normalized
output, then performs 24 timed serialization repetitions. Use the
same test file against the parent revision to compare, e.g. with perf
task-clock and Valgrind DHAT. The DHAT process totals include fixture
parsing and fixed-point verification; the printed elapsed time covers
only the repeated collector calls.
