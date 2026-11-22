# Lexical model visibility (#123)

The grammar needs declaration membership to distinguish passive model names
from scalar parameters, and the earliest model terminal from optional BJT
and MOS nodes. Model availability is lexical, not a global registry or
elaboration-time model selection.

Before ordered semantic parsing, the parser builds one structural index of
the expanded **card occurrences**. Each representable root/subcircuit body
has a local ASCII-case-insensitive set of declaration names, a set of local
MOS bin bases (`name.<ASCII digits>`), and its parent scope ordinal. The
ordered body-start keys bind semantic parsing to the corresponding occurrence
even when the resolver reuses the same underlying raw card in several
includes.

A grammar lookahead borrows the immutable visibility view: it searches the
local sets and then ancestor sets up to the root. It never clones the inherited
set, scans other scopes, or reparses a model declaration. A local bin base
does not validate the model's type or geometry; the model resolver still
chooses the backend and bin later. Numeric-looking model names and terminal
role errors retain the grammar's existing policy.

The index is deliberately non-validating. It sees valid lexical cards up to
the first `.end` or unmatched root `.ends`, skips lexical/source errors, and
tracks control sections and nested `.subckt`/`.ends` markers exactly as the
former read-only forward-name scan did. It does not tokenize an unselected
library section, expand an include, emit errors, or promote declarations from
body/control scopes. The semantic parser still reports each card's error in
original occurrence order, including malformed terminators, invalid control
scripts, and the 64-level nesting limit at the offending opener. No index
records are allocated beyond the nesting budget. Non-body grammar uses an
empty visibility view (include directives and evaluated measure cards).

For N card occurrences, the structural pass examines at most N cards once.
Storage is proportional to lexical bodies, unique local declarations and bin
bases, independent of the number of descendants that inherit them. Lookahead
cost is bounded by the lexical depth (64) times ordered-set lookup; binning
does not scan a range of similar names. Body lookup uses the ordered
occurrence keys. The AST, raw card references and include provenance are not
modified by this index.

Regression coverage includes differential comparison with the former
forward-name membership algorithm over malformed lexical streams, wide and
deep trees, nested ancestor/same-scope forward declarations, sibling and
child isolation, MOS bin base/suffix recognition, repeated included raw
card occurrences, exact nesting-limit diagnostics and writer round trips.
