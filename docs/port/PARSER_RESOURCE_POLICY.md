# Source admission and lexical work

Semantic parsing of borrowed input text, caller-owned folded decks and file-backed
include/library trees uses the same `SourceLimits` and per-operation
`SourceBudget`. The policy constrains source admission and lexical metadata,
separately from expression-tree limits and elaboration/device-expansion work.
Use `Parser::parse_text(path, text, limits)` for borrowed text (no include I/O),
`parse_file_with_limits_and_output(path, limits)` for real source trees, or
`parse_deck_with_limits_and_output(&deck, limits)` for an already-folded deck.
The ordinary semantic methods use defaults. `parser::classify_deck_with_limits`
applies the same admission and lexical counters to classification (all cards,
including after .end). `parser::load_classified_with_limits` and default CLI classification share one budget across loading, folding and classification; the low-level `source::load` also admits files before folding.

| Resource | Default | Charged |
|---|---:|---|
| Include depth | 64 | Maximum active include chain; configured depth cannot exceed 64 |
| File visits | 1,024 | Root and every include/section visit, including cached and empty selections |
| Source bytes | 16 MiB | Entire admitted source on each visit, including comments/unselected sections |
| Card work | 1,000,000 | Every ordered processed card occurrence, including repeated directives |
| Physical lines | 2,000,000 | Each loaded physical interpretation, including title, blank and comment lines |
| Folded cards | 1,000,000 | Each constructed logical card and corresponding possible lexical cache slot |
| Logical card bytes | 4 MiB | One folded card (including inserted continuation spaces) or title |
| Tokens | 4,000,000 | Each attempted lexical token, including punctuation and malformed final token, or the recorded count when replaying a cached lexical result |

A root reinterpreted as an included library fragment is a separate folding
interpretation. Repeated ordinary includes use the already folded immutable
fragment. Library indexing tokenizes only boundary directives, once when indexing
the fragment. Unselected ordinary cards have bounded folded storage but no
lexical objects or token charge. The cache preserves failed lexical results
as well as successful ones, but budget/allocation exhaustion is terminal and is
not cached as if it were malformed source. Replays spend the recorded token
count without creating new token storage.

Raw input files are read in bounded chunks: one excess byte is detected before
appending, and invalid UTF-8 is checked only after successful byte admission.
Folding charges lines/cards/combined text length before growing owned metadata
or continuation text. The tokenizer charges before allocating a new token or
its owned spelling/decoded string; lexer, card copies, cache slots and ordered
occurrence vectors reserve storage fallibly. Standard small object allocations
(Arc/path/error metadata, AST nodes) remain subject to their enclosing resource
counts; the policy is not a global allocator or exact RSS limit. Geometric
buffer capacity and allocator rounding may exceed admitted logical length.

Hard source-admission limits (bytes, physical lines, folded metadata, maximum
card length) apply to the entire input being admitted, even after .end or in an
unselected section; as with the existing whole-source byte limit, failure to
admit that input precedes semantic interpretation. Ordinary lexical errors and
exhaustion during ordered card/token traversal are replayed at their occurrence
so an earlier unsupported card retains its diagnostic. Semantic traversal stops
at .end; classification intentionally visits the full deck.

`source::parse_deck_text`, `fold_logical_lines` and `parse_fragment_text` are
low-level constructors of caller-owned folded input, not semantic admission
entrypoints. Their existing allocations are outside the parser's budget.
Admission of an externally constructed Deck charges its visible title and
logical text bytes, retained card records, and one physical line per card plus
the actual continuation metadata (or more if the public continuation count is
larger); the externally owned title counts as one retained line. It cannot
reconstruct discarded whitespace, comments or blank lines. Use borrowed-text
or file parsing when the original physical input itself must be bounded.

## Regression and measurement

`cargo +nightly test --test source_parse_limits --test sources` covers exact and
one-past limits, cached replay and lazy library boundaries, terminal lexical
budget errors and positioned continuations, 2,048-point independent-source PWL, 8,192-point behavioural PWL and 100k-device flat
decks. Existing parser, writer, structure and snapshot tests establish semantic
and provenance parity. The ignored benchmarks in source_parse_limits use one
million punctuation tokens or one million tiny cards with an 8192-element budget, plus a fully admitted punctuation baseline;
run each separately on the test executable with `--exact --ignored --nocapture`
under time and DHAT to measure actual retained heap and time.
