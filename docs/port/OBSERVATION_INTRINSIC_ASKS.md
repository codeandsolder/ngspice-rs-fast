# Intrinsic device asks versus terminal-current observations

C SPICE instance asks are not necessarily physical currents entering the
external terminal of the device. MOS intrinsic drain current, BJT collector
current, normalized JFET/MESFET channel and gate currents, and BSIM4 body
and gate leakage have distinct definitions from the external MNA residual.

## Bound observation contract

| Query | Implementation | Meaning |
| --- | --- | --- |
| i(device) | Existing branch column or isolated terminal residual | Physical current at first terminal |
| @device[i] or @device[current] | Isolated terminal residual | Physical current at first terminal |
| @device[p] or @device[power] | Isolated residual times terminal voltage | Two-terminal external power |
| @device[dc] for V/I sources | Effective source forcing, including sweep override | Source value |
| Any other @device[key] | Device-specific parameter/operating ask | Original C device ask or explicit unsupported |

A designator/keyword heuristic must never reinterpret C intrinsic fields
such as id, ic, ib, ie, ig, is, cd, cg or cgd as the current at an external
pin. Device-specific model ask implementations are authoritative. Unsupported
intrinsic fields produce a structured not-yet-ported error, never a silently
substituted terminal residual, a guessed zero, or a sign adjustment.

The terminal-current path retains the source-specific sign conventions and
alias/ground handling, and a single isolated stamp per selected load per
point. Explicit physical current and power observers share the same
prepared scratch. Selected intrinsic-only queries do not allocate a terminal
current buffer or stamp a device. Replacement model contexts, topology
identity, state isolation and epoch-based observation invalidation remain
unchanged.

## Model semantics

BSIM3 and BSIM4 implement model-specific channel current id; BSIM4 draws
from its C-referenced intrinsic small-signal cd rather than physical drain
residual, particularly when gate tunnelling or an internal body network
matters. JFET and JFET2 intrinsic id, ig and is use the exact C-style
loaded cd and cg, including gate junctions, before polarity normalization
is applied to external terminal currents. MESFET cd, cg and is similarly
use its loaded Statz channel and gate currents. HFET1 and MESA already
implement their own C-backed cd, cg and cs asks.

The diode, Gummel–Poon BJT, and shared classic MOS level implementations
now provide their C-specific DC intrinsic asks. Diodes evaluate DIOcurrent
(`@d1[id]`/`@d1[c]`), intrinsic `vd` and `gd` from the
temperature-adjusted junction with gmin and internal series node. The BJT
returns `BJTcc` and `BJTcb` as `@q1[ic]` and `@q1[ib]`, with the
multiplicity and lateral-substrate correction prescribed by `bjtask.c`.
Classic MOS levels replay `MOSXcd = mode*cdrain - cbd` for `@m1[id]`,
respecting each level's multiplicity placement. These quantities are not
external terminal residuals: use `i(m1)`, `@m1[current]`, `i(q1)` or
`@q1[current]` explicitly for terminal current.

For the independent C ngspice-47+ BJT deck
(`vcc=5 V`, `vb=0.7 V`, `is=1e-14 A`, `bf=100`),
`@q1[ic] = 0.0056703467757422835 A` and
`@q1[ib] = 5.670346410422285e-5 A`. The external collector terminal
current differs by about `5e-12 A` owing to the physical gmin branch.
The integration test checks both reference quantities and that difference.
Unsupported model-specific asks still fail explicitly; an external residual
is never silently substituted.

State-backed transient C asks, including JFET/MESFET currents that depend
on integrated gate charge, remain explicitly unsupported until the required
accepted-state values can be observed with correct time semantics.

## Reproducible verification

The resolver's focused internal tests verify that selecting C intrinsic
asks uses no isolated device stamp and no initialized terminal-current
scratch; explicit terminal current/power selectors share one load. The
new tests/intrinsic_current_asks.rs checks a committed BSIM4 ngspice
operating-point deck with tunnelling, body network and multiple devices.
Independent C ngspice-47 values for this deck include m1 id
2.743483712211524e-4 A, gm 8.464358463227711e-4 S and ibd
-1.300009999988938e-12 A.

The JFET, JFET2, MESFET, HFET1 and MESA integration tests cover
cross-family device asks, both model polarities and original numerical
reference equations. All source-value tolerances, source/terminal
signs and C numerical comparison criteria remain unchanged. The serial
campaign validator runs full nightly lint and the C golden corpus.

Related specifications: SOURCE_BINDING.md and
OPTIMIZATION_OBSERVATION_WORKSPACE.md. The issue-specific Notion worklog
holds the dated implementation history, experiments and test evidence.
