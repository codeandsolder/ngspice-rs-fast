# BSIM4 OP/DC C-oracle numerical diagnosis — issue #154

## Status (2026-11-23)

Four independent `tests/bsim4_device.rs` cases exceed the pre-existing C
comparison bound `|Rust - C| <= 1e-9 |C| + 1e-30`. The source-deck and C
reference values are unmodified. All four failures reproduce on the current
campaign base `bffba1f8b250f9b9c6e5df0e4e61ae8467cf1d54`, before any
changes to the BSIM4 physics or solver. This is a diagnostic report, not a
claim that the four strict oracle tests have been repaired.

| Test and point | C | Rust | Absolute difference | Relative difference |
| --- | ---: | ---: | ---: | ---: |
| PMOS `rdsmod=1`, `i(vd)[1]` | -4.144328543537190e-12 | -4.144328697065342e-12 | 1.535282e-19 A | 3.705e-8 |
| NMOS `@m1[w]` sweep, `i(vb)[0]` | 1.202395821797708e-12 | 1.202394954435030e-12 | 8.673627e-19 A | 7.214e-7 |
| Temperature sweep, `i(vg)[4]` | -8.484372804318241e-8 | -8.484372813372282e-8 | 9.054041e-17 A | 1.067e-9 |
| NMOS OP, gate/body network, `i(vb)[0]` | 1.204238098129196e-12 | 1.204240353268773e-12 | 2.255140e-18 A | 1.873e-6 |

The source test loop applies no tolerance changes. Its five other BSIM4
device cases also remain part of issue acceptance, along with sensitivity and
committed goldens.

## Reproducing exact C source and internal asks

The installed `/usr/bin/ngspice` is ngspice 47, with a KLU-capable build.
For the amplifier OP, PMOS DC and temperature DC decks, running
`ngspice -b -r result.raw deck.cir` with `.options sparse` yielded
**bit-for-bit identical** observed current values to the existing fixed C
goldens. A C width sweep with `.dc @m1[w]` is not accepted by that binary,
so the same parameter values were run as four independent `.op` decks with
literal instance widths. The first independently solved C OP at `w=0.3u`
reproduced the test's width-sweep index-zero C current bit-for-bit:
`1.2023958217977082e-12` A. The same `.op` in Rust reproduced its existing
`1.2023949544350304e-12` A. This isolates the discrepancy from the
sweep driver's warm-start/temperature updates at that point.

For C/Rust physical comparisons, add
```
.save all @m1[id] @m1[gm] @m1[gds] @m1[gmbs] @m1[ibd] @m1[ibs] @m1[igs] @m1[igd] @m1[igb] @m1[igcs] @m1[igcd] @m1[isub] @m1[igidl] @m1[igisl] @m1[vth] @m1[vdsat] @m1[vgs] @m1[vds] @m1[vbs]
```
and save binary rawfiles from both executables. The C deck also has
`.options sparse`; **remove only this option for the Rust CLI**, which does
not accept the C compatibility selector and always uses its own sparse
solver. The C and Rust binary rawfile headers use different vector spellings
for device asks (`i(@m1[ibd])` versus `@m1[ibd]`), which need canonicalization
when comparing columns.

Relevant Acer diagnostic data and scripts are in
`/home/jan/build/ngspice-rs-fast/workers/arch-1122-2154-b6e9/issue154-diagnostics`.
`compare_pairs.py`, `all_diffs.py`, `compare_raw.py` and the rawfiles
record every available intermediate measurement. These are per-issue working
artifacts, not repository-global build products. The C oracle source is
`/home/jan/projects/ngspice/src/spicelib/devices/bsim4/{b4ld.c,b4temp.c}`;
the pinned reference source is under
`/home/jan/build/ngspice-rs-fast/parallel-analysis/pinned-c/src`.

## Intermediate results

All four failures have almost identical physical device preparations and
small-signal quantities at the compared bias. For the amplifier operating
point, the C and Rust voltages at `m1#gate`, `m1#body`, `m1#dbody` and
`d` are bitwise equal; `m1#sbody` differs by only one binary64 ULP.
The physical C/Rust `@m1[ibd]` values are bitwise identical,
`@m1[ibs]` differs by about `1.26e-16` relative,
`@m1[igm]` is not requested, and `@m1[gm]`, `@m1[gds]`,
`@m1[gmbs]`, `@m1[igb]`, `@m1[isub]`, `@m1[igcs]`
and `@m1[igcd]` differ at roughly binary64 rounding levels.
Nevertheless the external source current `i(vb)` differs by
`2.255140e-18` A, or `1.873e-6` relative to a current near `1.2e-12` A.

The width `w=0.3u` OP exhibits the same pattern: `@m1[ibd]` and
`@m1[ibs]` agree within 1–2 ULPs, with `@m1[gm]` and `@m1[gds]`
agreeing better than 1e-14 relative; yet the **solved voltage-source
current** differs by `8.674e-19` A. For the temperature 20°C point,
`@m1[gm]` and `@m1[gds]` agree at about 1e-15 relative while `i(vg)`
differs by 9.05e-17 A.

There is a **separate discrepancy** in the interpretation or stage at which
`@m1[id]` is obtained: on the amplifier OP, the C ask reports
`3.482559231937092e-8` A, while the Rust ask reports
`2.153518804410319e-8` A (very close to the external drain-source branch
current). Its value differs substantially while `@m1[gm]` and terminal
voltages match. The C `b4ask.c` `BSIM4_CD` asks the stored
`BSIM4cd`; investigate the live/previous Newton load stage and the
`observation_operating` recomputation in
`src/devices/bsim4/device.rs` separately from source-current residuals.
This needs a dedicated test/source reservation rather than a guessed
current correction.

### One-ULP sensitivity experiment

At the *unchanged original physical model*, modify **only** the DC bias
source literal `vb b 0 dc -0.1` by one neighboring representable binary64
value (either `-0.10000000000000002` or `-0.09999999999999999`),
keeping all model parameters and oracle observations unchanged.

For the width `w=0.3u` OP, Rust's `i(vb)` changes from
`1.2023949544350304e-12` to `1.2023958217967683e-12` A for the
lower neighboring input — almost exactly the old C golden
`1.2023958217977082e-12` (residual difference `9.40e-25` A).
For the amplifier, the C output `i(vb)` itself changes by
`6.938894e-18` A for the lower neighboring bias input; this is several
times larger than the entire Rust/C mismatch and over 5,000 times the
test's relative `1e-9` current bound. The Rust result also changes by
`1.734723e-18` A for that same one-step input perturbation.

These intentionally perturbed decks are **diagnostics only**; they must
not replace or alter a C golden or production bias. The result shows that
the near-balanced source-current solves are sensitive to differences at
the scale of a single representable source voltage. Independent C
SPARSE/KLU runs on the tested decks happened to produce identical
source-current bits, so this observation alone does not establish a
factor-order discrepancy between those backends.

## Consequences and next investigation

1. The model coefficients, temperature-dependent junction currents and
   physical gate leakage already agree closely; avoid changing model
   equations to force a particular external voltage-source current.
2. Analyze the **last Newton matrix, right-hand-side and source-current
   solution residual** in the body/gate resistor network, including
   cancellation of common-mode voltages and pivot/back substitution.
   A physically improved residual should be measured against an
   independently higher-precision solution before being asserted as
   closer to either old C source current.
3. Investigate the independent `@m1[id]` live-record mismatch; it may
   reveal a valid state/observation bug despite the current source
   discrepancies.
4. Retain the exact C fixture and strict `1e-9` comparison policy until
   the underlying numerical difference is resolved. Neither raising
   tolerance nor shifting the bias source by a ULP is an acceptable fix.



### Direct reconstruction of the body-source resistor network

The C `b4ld.c` external `B` row and the Rust `load/stamp.rs` external
body row contain three **linear** branches: `B↔DB` at `grbdb`, `B↔BP`
at `grbpb` and `B↔SB` at `grbsb`. Both fixture model cards give
`rbdb=15 Ω`, `rbpb=5 Ω`, `rbsb=15 Ω`, with unit multiplicity. This
permits reconstruction from **only the saved double-precision node voltages**:

\[
I_{vb,\mathrm{rounded}} =
\frac{V_{DB}-V_B}{15}\;+
\frac{V_{BP}-V_B}{5}\;+
\frac{V_{SB}-V_B}{15}.
\]

The next table uses the exact binary64 bit patterns for all saved voltages,
converted to 80-digit decimal before evaluating the differences and resistor
quotients. The reconstructed value is not a replacement for the solver result:
source-current equations are eliminated with their own intermediate rounding,
and the saved node voltages have already lost a few low bits.

| Case | Solver | `i(vb)` in rawfile (A) | Reconstructed from rounded nodes (A) | `i(vb)-reconstructed` (A) |
| --- | --- | ---: | ---: | ---: |
| amplifier OP | C | 1.20423809812919558e-12 | 1.20424041109383021e-12 | -2.312964634635743e-18 |
| amplifier OP | Rust | 1.20424035326877254e-12 | 1.20424133627968407e-12 | -9.830109115267104e-19 |
| width 0.3 µm OP | C | 1.20239582179770821e-12 | 1.20239651568709860e-12 | -6.938893903907228e-19 |
| width 0.3 µm OP | Rust | 1.20239495443503035e-12 | 1.20239559050124475e-12 | -6.360662143926444e-19 |

For the amplifier, the C and Rust saved `V_DB` and `V_BP` bit patterns
are **identical**, and `V_SB` differs by precisely one ULP
(`1.3877787807814457e-17` V). This contributes exactly
`9.2518585385e-19` A to the difference in the reconstructed branch
current. The remaining `1.3299537e-18` A of the measured
`2.25514e-18` A source-current discrepancy follows from the distinct
rounded source-current equations. For the width fixture the same single
`V_SB` ULP explains `9.2518585e-19` A of the observed
`8.6736268e-19` A mismatch; the residual terms partially cancel.

This is direct evidence of a **floating-point cancellation/solve-observation
problem** rather than inaccurate BSIM4 junction, gate or temperature physics.
It does not certify the correctness of either source-current rounding path.
An independently high-precision solve is needed to determine the more
accurate voltage-source current before changing the solver. The 1e-9 C
fixture bound stays unchanged.

## Independent high-precision MNA solve checkpoint

The opt-in bsim4_op_diagnostics.rs tests export the final Rust
Newton-linearized MNA matrix by per-device (row,column,value) triplets,
RHS, converged vector and the post-load next iterate for both amplifier
and width=0.3 um cases. Script
/home/jan/build/ngspice-rs-fast/workers/arch-1122-2154-b6e9/issue154-highprecision-mna.py
reassembles those entries with 90-decimal-digit arithmetic, computes
Gaussian elimination with partial pivoting and independently evaluates
the decimal residual of each saved binary64 solution.

The reconstructed body-source currents (row 8) are:

| Case | Original Rust (A) | Independent 90-digit solve of rounded MNA entries (A) | Difference (A) |
| --- | ---: | ---: | ---: |
| Amplifier | 1.20424035326877254e-12 | 1.20424640237558771e-12 | +6.049107e-18 |
| Width = 0.3 um | 1.20239495443503035e-12 | 1.20240053220768701e-12 | +5.577773e-18 |

For the original saved Rust solution, evaluating the *rounded* MNA
matrix in 90-digit arithmetic gives body-row residuals
-9.8302444381e-19 A (amplifier) and -6.3607973273e-19 A (width).
Maximum residuals across all rows are 1.6387289894e-17 and
2.2102437895e-17, respectively. The 90-digit solution has residuals
around 1e-91 A, as expected for this 11x11 system.

**Important**: a mathematically more accurate solve of Rust's last
linearized matrix does NOT reproduce the fixed C source-current golden;
it moves the answers even farther from C. A different Newton convergence
endpoint or model iteration state may be relevant. Do not replace the
solver with an arbitrary correction chosen to fit a near-cancelled
current, and do not relax the existing strict oracle bound. The
high-precision solve also does not represent the full nonlinear solution
of f(x)=0; it only solves J(x) x_next = rhs for the last available
linearization.
