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

## Independent high-precision check of the last Rust Newton system

An opt-in diagnostic in tests/bsim4_op_diagnostics.rs replays an
unlimited MODEINITFLOAT DC load at the Rust converged iterate, retaining
the exact assembled sparse triplets, right-hand side, source-current
unknowns, and saved point. Its output was analyzed by a 90-decimal-digit
partial pivot Gaussian solver using exact Decimal.from_float
representations of the printed binary64 coefficients and RHS. The
analysis is stored in the separate campaign work directory under
build/arch-1123-g6c9-154/diagnosis/high_precision_mna.txt.

| Bias | Rust saved i(vb) | Exact solve of last Rust A*x=b | Fixed C oracle i(vb) |
| --- | ---: | ---: | ---: |
| NMOS amplifier OP | 1.20424035326877254e-12 | 1.20424634826971557e-12 | 1.20423809812919560e-12 |
| Width=0.3um OP | 1.20239495443503035e-12 | 1.20240326913628876e-12 | 1.20239582179770820e-12 |

The final sampled matrix has a tiny but nonzero nonlinear residual:
e.g. the amplifier body-source-voltage row residual is
-9.83024444e-19 A and the width case is -6.36079733e-19 A.
Re-solving the same rounded Rust Jacobian/RHS exactly does NOT
approach the C oracle, and is further from it than the original saved
Rust current in both fixtures. This is not evidence that either
solver is mathematically more accurate: the matrix is a subsequent
linearization with state and coefficient roundoff, not a
high-precision nonlinear evaluation at its exact root.
It rules out using a higher-precision solve of this one frozen
matrix as a plug-in fix for all four C-current mismatches.

The fixed gate bias illustrates the intrinsic parity problem:
C saves v(g) = 0.70000000000000006661 whereas Rust saves
v(g) = 0.69999999999999995559, despite the identical binary64
literal 0.7. C's result is one binary64 ULP above the exact source
constraint while Rust's is the input itself. Tight relative
source-current parity for differences of ~0.1-V nodes cannot
distinguish normal LU differences at an external body current near
1e-12 A. Do not perturb the input or raise the numeric tolerance.

## Channel-current observation mismatch is independent

C b4ask.c:BSIM4_CD returns physical BSIM4cd (channel current),
not the external drain-source branch current after gate tunneling
and junction contributions. The saved amplifier C snapshot satisfies
channel-current conservation:

- C @m1[id] = 3.48255923193709199e-8 A.
- C external drain current -i(vdd) = 2.15351880440608351e-8 A.
- Gate/junction corrections @m1[igcd] + @m1[igd] + @m1[ibd] - @m1[isub]
  supply the missing ~1.3290404275e-8 A.
- Independent reconstruction from saved binary64 inputs yields
  3.48255923193289619e-8 A, only 4.20e-20 A from C's channel ask.

Rust's @m1[id] instead reports 2.15351880441031867e-8 A,
essentially the external drain branch, while its separate gate
and junction asks match C closely. At the 20C temperature point,
C channel 5.494453808338901e-4 A versus Rust ask
5.494269230159224e-4 A (near the external 5.494269274384359e-4 A).
This is a distinct observation-binding issue, now pinpointed below.
It does not explain the ~1e-18-A external body-source current
discrepancies; do not alter the independent oracle values.


## Confirmed root of the independent BSIM4 channel ask discrepancy

A temporary per-key trace in the reserved BSIM4 device source showed both
the live instance and a fresh MODEINITSMSIG observation replay carry
h.cd = 3.48255923193709464e-8 A. That matches C channel current
3.48255923193709199e-8 A within 2.65e-23 A (relative ~7.6e-16).
The simultaneous gm observations match to the same rounding level.
Neither model evaluation nor replay loses the channel-current value.

Yet the exported @m1[id] remains 2.15351880441031867e-8 A and
the per-key trace prints no device-specific observation for id.
The reason is the general observation binder
(src/devices/observe.rs): terminal() maps designator m + key id
to terminal slot zero before the model-specific ask lookup. This
routes the alias to isolated terminal residual-current reconstruction
and bypasses the correct BSIM4_CD field entirely. The same shortcut
covers j/id, q/ic and z/cd, although their C device asks can have
different physical definitions, including junction and gate currents.
Only explicit i(device) and @device[i]/[current] are unambiguous
generic terminal-current requests.

Fixing this requires an observation selector contract that chooses
device-specific C asks when defined, *not* a BSIM4-only return-value
correction and not an arbitrary blanket conversion of model asks into
terminal currents. Preserve proper fallback semantics for the remaining
device families with independent C reference tests, and retain the
ability to request raw terminal current explicitly. This is
independent of the four tiny external-voltage-source-current
numerical parity issues in #154.

The temporary BSIM4 source trace is diagnostic-only and was removed
after recording its results. Read-only opt-in fixture coverage remains
in tests/bsim4_op_diagnostics.rs.

## Exact solver replay on actual Newton inputs (2026-11-24)

On integration base `00eb948bdfdf231a127ea4b04dae8ac2e4969b18`,
all four original failing tests reproduce unchanged. The earlier
"possible pivot order" diagnosis is now resolved for these specific
failure points: the source-current discrepancies can be reproduced
without any model reload, parameter change or new nonlinear iteration.

Temporary diagnostics captured each real `ni_step`'s bound pattern
in allocation order, binary64 coefficients and RHS, reorder flag,
diagonal gmin, fresh-frame indicator, factor permutation, raw LU values
and returned solution. The production sources are not the permanent
diagnostic interface. The independent replay linked the existing
pinned C build's original `spalloc.o`, `spbuild.o`, `spextra.o`,
`spfactor.o`, `spoutput.o`, `spsolve.o`, `sputils.o` and
`misc/alloc.o` (compiled with `-O2 -ffp-contract=off`).
It created elements in recorded order, preserved the factor frame
between steps, reset on fresh frames, ran `spMNA_Preorder` and
`spOrderAndFactor` or `spFactor` as appropriate, then `spSolve`.
No C device model or convergence loop was involved.

| Case | Rust source current (A) | C SPARSE, same Rust inputs (A) | Fixed C fixture (A) |
| --- | ---: | ---: | ---: |
| Amplifier OP | 1.2042403532687725e-12 | 1.2042380981291956e-12 | 1.2042380981291956e-12 |
| Width sweep point 0 | 1.2023949544350304e-12 | 1.2023958217977082e-12 | 1.2023958217977082e-12 |
| PMOS sweep point 1 | -4.144328697065342e-12 | -4.14432854353719e-12 | -4.14432854353719e-12 |
| Temperature sweep point 4 | -8.484372813372282e-8 | -8.484372809869356e-8 | -8.484372804318241e-8 |

At the failing iterations the pivot permutations match exactly,
and **every factor** matches C numerically exactly after converting
the Rust raw factors to C's inverse-diagonal/normalized-upper
representation (40/40 amplifier, width and temperature; 28/28 PMOS).
The amplifier's four Newton passes also all match in factors.
The temperature replay differs from the fixture by 5.551115e-17 A,
inside the unmodified 8.484373e-17 A acceptance bound.
Thus all four initial failures have a substitution-rounding component
large enough to explain the failure, independently of model assembly.

The numerical mechanism is the intentional raw-factor implementation:
Rust uses raw numerators in both triangular scatter directions and
recovers cross-product cancellation; C rounds the normalized factors,
scatters the first substitution and gathers the second from the
already rounded intermediate vector (`spsolve.c`).
The two operations are algebraically identical but not bit-equivalent.

An independent 100-decimal-digit solve of the exact *actual* input
coefficients and RHS at each failing iteration (not a later linearization)
gives the following comparison. Amplifier and width use the third
iteration matching the failing current; PMOS uses iteration seven,
temperature iteration twelve.

| Case | Exact rounded-system solution (A) | Rust minus exact (A) | C replay minus exact (A) |
| --- | ---: | ---: | ---: |
| Amplifier | 1.20424640235956376733e-12 | -6.04909e-18 | -8.30423e-18 |
| Width | 1.20240053220584279507e-12 | -5.57777e-18 | -4.71041e-18 |
| PMOS | -4.14432879652489118225e-12 | 9.94595e-20 | 2.52988e-19 |
| Temperature | -8.48437280998981750298e-8 | -3.38246e-17 | 1.20461e-18 |

Neither arithmetic path always gives the more accurate solution of the
rounded matrix. The original issue's strict fixture acceptance and the
documented native solver's cancellation-recovery contract are distinct
requirements, and should not be conflated with the exact nonlinear
physical solution.

Scratch reproduction on Acer:
`/home/jan/build/ngspice-rs-fast/quality/issue-campaign-20261120/research/arch-1123-2317-k9d2-154/`
contains `replay.c`, `compare.py` (run via uv), raw input/replay files
for the four tests and `all-four-replay-report.txt`. Rust traces are
in the sibling campaign build directory `arch-1123-2317-k9d2-154`.
Temporary source instrumentation is preserved externally as
`rust-trace.patch`; do not apply it to normal builds.

### Substitution variants

Separating the two triangular passes in the captured systems shows
that the robust Rust forward scatter can remain intact: with either
forward arithmetic, normalized backward gather reproduces the three
exact C fixture values and places temperature within the original
tolerance. Gathering the raw upper factors before division, or
scattering normalized terms, is insufficient across all four cases.

A temporary direct C-order real normal solve passed all nine BSIM4
device tests, with no fixture changes. However, three existing
SPARSE unit tests rejected the implementation: reciprocal range,
unrepresentable normalized upper entries, and scalar/batched bit
agreement. The raw factors and recovery kernels are therefore
essential; restoring the entire old algorithm is not a valid fix.

The guarded-gather experiment keeps those factors, preserves the
robust forward pass, and tests normalization range and a local
absolute roundoff bound for each backward row. It retries the whole
RHS through raw scatter if a row's sign/zero cannot be resolved
within that bound. This is a local arithmetic check, not a condition
number estimate or a guarantee of physical forward accuracy.
Independent Python replay of the guard over all 125 captured Newton
systems (4 amplifier, 10 width, 83 PMOS, 28 temperature) accepts
every system and yields the same four C-compatible results. The
script is `guarded-replay.py` in the scratch directory above.
