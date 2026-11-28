# LTspice 24.0.12 sampled-current evidence (#274)

Independent Windows LTspice batch ASCII raw runs (2026-11-28). The three
included decks are the original tight-tolerance feedback, isolated sine,
and isolated pulse witnesses. `ltspice-24.0.12.tsv` summarizes all 26 runs.

Reproduce the matrix by varying `method=trap|gear`, `reltol=1e-3|1e-6`,
and feedback gain `0|.001|.05|.5` in the feedback deck (16 runs), plus the
original feedback gain .5 at `reltol=1e-4` (2 runs, named `trap` and `gear`).
The two isolated decks each use both methods and both tolerances (8 runs).
Run with `LTspice.exe -b -ascii <deck>` and `plotwinsize=0`.

The error columns compare the behavioural monitor V(b) with I(C1) at the
preceding and present raw output points respectively (excluding the initial
point). The endpoint columns contain the last time, V(s), V(p), and I(C1).
I(R1) independently checks the physical terminal current. The raw display
points are not assumed to be the exact Newton iteration history.

The tolerance dependence and disagreement with preceding output currents
falsify a universal accepted-step hold. The evidence supports a sampled
iteration policy, but does not identify LTspice's precise internal iteration
algorithm. The native regression uses the same physical network and excitation,
with explicit phase-selected control; integration methods need not produce
identical adaptive output grids.

A separate strong-feedback check varied the feedback coefficient to 5, 10,
20 and 100 ohms, retaining the tight trapezoidal deck. All four failed with
LTspice's `Time step too small` error (gain 5 reached approximately 4 ps;
10, 20 and 100 failed at the initial transient point). This is consistent
with convergence limitations of an iteration-sampled feedback law. It is
not a requirement that the native solver reproduce vendor failures.
