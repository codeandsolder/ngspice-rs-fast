# Bounded model-backed passives (#19, #20)

Model-backed passive inputs are projected into typed, immutable setup and
temperature state. Ordinary loads, DC replacements and sensitivity replay
share the C-ordered arithmetic in `devices/passive/physics.rs`; sensitivity
still owns C's ordered setter/replay records and can visit unbounded perturbed
states. Validation and diagnostic attribution remain separate from arithmetic.

## Supported surface

The table is exhaustive: other model/instance setters error, even if a supplied
scalar would make their geometry branch unnecessary. Setters remain raw and
ordered in the AST; each scalar is range-checked, then the last duplicate wins.
The first model declaration wins case-insensitively, as in `INPmakeMod`.

| Family | Model setters | Instance setters |
| --- | --- | --- |
| R (`r`, `res`) | `r`, `rsh`, `defw`, `l`, `narrow`, `short`, `tc1`, `tc2`, `tce`, `tnom`, `kf`, `af`, `ef`, `lf`, `wf` | `resistance` (parser maps `r`), `l`, `w`, `tc1`, `tc2`, `tce`, `temp`, `dtemp`, `ac`, `scale`, `m`, `noisy` |
| C | `cap`, `cj`, `cjsw`, `defw`, `narrow`, `short`, `del`, `di`, `thick`, `tc1`, `tc2`, `tnom` | `capacitance` (parser maps `c`/`cap`), `l`, `w`, `tc1`, `tc2`, `temp`, `dtemp`, `scale`, `m`, `ic` |
| L | `ind`, `csect`, `dia`, `length`, `nt`, `mu`, `tc1`, `tc2`, `tnom` | `inductance` (parser maps `l`), `nt`, `tc1`, `tc2`, `temp`, `dtemp`, `scale`, `m`, `ic` |

`level` is separately consumed by the existing resolver's bounded selector
policy, not ignored as a physics setter; see [MODEL_SCHEMAS.md](MODEL_SCHEMAS.md).
This expanded instance surface requires a declared model. Literal factory/API
support is unchanged (primary scalar, resistor noisy and C/L IC); negative nonzero scalar
instance R remains allowed. Model R/C/L setters must be positive; explicit
instance C/L may be zero. The AC resistance is independently nonzero.

Units: R/RSH in ohms (RSH per square); C in farads; L in henries; geometry in
metres; CJ in F/m²; CJSW in F/m; TC1 in K⁻¹; TC2 in K⁻²; TEMP/TNOM input in
Celsius, DTEMP in kelvin. TCE is a per-kelvin exponent coefficient.
Coil dimensions are in metres (cross section in square metres);
NT, MU and DI are dimensionless. Scale and multiplicity are positive finite dimensionless scalars, not
integer-only. C IC is volts; L IC is amperes. Under companion `.tran ... uic`
the instance IC seeds the model-backed C/L exactly as a literal one (charge
from the effective, temperature-adjusted capacitance; inductor branch current),
and the model-backed C/L charge/flux takes part in truncation control like the
scalar device it delegates to (#80 review fix; checked against C for coupled
inductors and a TC/`m` capacitor). Without `uic` IC is ignored, as in C.

## Values, geometry and defaults

### Resistor

References: `ressetup.c::RESsetup`, `restemp.c::RESupdate_conduct`.

1. Last explicit instance resistance wins over model/geometry.
2. Otherwise positive RSH selects sheet geometry **ahead of** model R, even
   when L/W are omitted (C supplies default dimensions).
3. Otherwise explicit model R is required; Rust does not use C's warning-and-
   1 mOhm missing-resistance fallback.

```text
L = instance l, else model l, else 10 um
W = instance w, else model defw, else 10 um
Leff = L - 2*short
Weff = W - 2*narrow
Rbase = (Leff / Weff) * RSH
```

RSH defaults to zero (geometry disabled); short/narrow default to zero. RSH and
corrections must be nonnegative; L/W and effective dimensions must be positive.
The nominal value must be nonzero with representable finite reciprocal. Negative
model R is rejected (C ignores that setter); negative instance R is retained.
The explicit instance R setter clamps tiny magnitude to C's RESMIN.
The model R setter remains bounded. Global frontend geometry scale is
applied once to explicit instance R/C length and width, not model defaults. C's sheet-geometry eligibility tests the uncorrected L*W*RSH product;
Rust additionally checks effective dimensions before accepting the result.

### Capacitor

References: `capsetup.c::CAPsetup`, `captemp.c::CAPtemp`.

1. Last explicit instance C wins.
2. Otherwise explicit model CAP wins over geometry.
3. Otherwise require positive instance L, and use instance W or model DEFW
   (default 10 um).

```text
Leff = instance L - short
Weff = W - narrow
Cbase = CJ*Weff*Leff + CJSW*2*(Leff+Weff)
```

CJ/CJSW and corrections default to zero and must be nonnegative; effective
L/W and final C must be positive. **C uses one correction, R uses two.**

C stores model `defl` but does not apply it to instances: `CAPsetup` initializes
missing instance length to zero. The live setup probe exposed this distinction.
Rust therefore rejects `defl` and missing/degenerate L, rather than inventing a
model-length default or accepting C's negative-area/sidewall corner cases.
At setup, an explicit CJ wins; otherwise positive THICK derives CJ from
DI times C's vacuum permittivity (default silicon dioxide when DI is absent).
DEL supplies twice its value to each missing NARROW/SHORT independently.
The prepared geometry coefficients are immutable across ordinary temperature
and instance-sweep evaluations; instance L/W are temperature-time inputs.
Both C and Rust's sensitivity replay recompute the capacitance from those
inputs, with C's left-to-right product grouping.

### Inductor

References: `indsetup.c::INDsetup`, `indtemp.c::INDtemp`, `indacld.c::INDacLoad`.
Last explicit instance L wins; otherwise written instance NT uses the
setup-prepared specific inductance times NT twice, in that order. Without
instance NT, use explicit model IND, or the model NT squared times specific
inductance. DIA takes precedence over CSECT, and the prepared single-turn
inductance is MU times vacuum permeability times cross section divided by
LENGTH, with C's Lundin correction and its small-geometry cutoff. The normal
path requires positive complete coil dimensions and turns when deriving L;
it does not use C's missing-inductance zero fallback. Instance NT sweeps
reuse the setup result without rerunning Lundin.
K cards couple model-backed inductors through C's `INDinduct` (temperature- and
scale-adjusted, before `/m`); see [MUTUAL_INDUCTANCE.md](MUTUAL_INDUCTANCE.md).

## Temperature, scale and multiplicity

References: `restemp.c`, `captemp.c`, `indtemp.c`, `capacld.c`, `indacld.c`.

```text
T = instance TEMP, else AnalysisContext.temperature
DTEMP = 0 when TEMP was given, else written DTEMP or 0
Tnom = model TNOM, else AnalysisContext.nominal_temperature
DeltaT = ((T+273.15) + DTEMP) - (Tnom+273.15)
TC1 = instance TC1, else model TC1, else 0
TC2 = instance TC2, else model TC2, else 0
f = 1 + TC1*DeltaT + TC2*DeltaT²
Rscaled = Rbase*f*scale
Gdc = m / Rscaled
Gac = m / (ACbase*f*scale)  # ACbase = explicit AC or Rbase
Reffective = Rscaled/m
Ceffective = Cbase*f*scale*m
Leffective = Lbase*f*scale/m
```

Scale/M default to 1. TC1/TC2 are independent per-coefficient overrides; explicit
zero overrides a nonzero model coefficient. TEMP does not change TNOM. R uses
C's Horner polynomial evaluation order; C/L use the direct sum. A written
resistor TCE (instance over model, including explicit zero) selects
`pow(1.01, TCE*DeltaT)` instead of the polynomial. Normal stamps and sensitivity
replay use conductance directly, not the reciprocal of rounded Reffective.
AC and pole-zero use ACbase; DC/transient/noise current use Rbase. AC reactive
stamps multiply by frequency before applying M, as C does. All temperatures
must be finite and above absolute zero; factors must be positive and finite.
Derived overflow, zero effective values and nonfinite conductance error explicitly;
no cumulative correction can conceal them.

`CAPask` includes M in its capacitance query; `INDask` omits M in its inductance
query. The oracle divides C's queried inductance by M before comparing the value
actually stamped. R's queried resistance is unadjusted; with a 1 V drive its
queried current verifies effective conductance instead.

## Production APIs and atomicity

- `ResolvedModel::passive_parameters(&instance)` returns immutable typed
  `PassiveParameters`: `family`, `nominal_value`, `multiplicity`,
  `initial_condition`, and `effective_value(&ModelContext)`.
- `Circuit::from_netlist` validates at 27 Celsius. For a nondefault construction
  temperature use `from_netlist_with_context(&netlist, &context.model_context())`.
- All four analysis drivers copy `AnalysisContext` settings via `model_context`
  into `Circuit::linear_system_with_context`. Default `linear_system()` remains
  the 27 Celsius convenience API.
- `LinearContext` carries explicit `ModelContext`; `StampContext` now carries
  both circuit and nominal temperatures. The wrapper derives scalar R/C/L and
  delegates to existing stamps, sharing C-ordered AC stamp kernels with
  sensitivity replay. Recipes are immutable across hot/cold/repeated
  analyses; factors, node IDs, branch order/signs and accepted state are preserved.
- Schema/geometry/constructor failures leave the caller's circuit/node table
  unchanged. Unused model declarations still error rather than discarding physics;
  duplicate shadowed declarations follow the documented first-definition policy.

Example through the analysis API:

```rust
let context = analysis::AnalysisContext {
    temperature: 77.0,
    nominal_temperature: 27.0,
};
let mut circuit = devices::Circuit::from_netlist_with_context(
    &netlist, &context.model_context(),
)?;
let plot = input.resolve()?.run(&mut circuit, &context)?;
```

## Explicit limits and verification

Behavioral/nonlinear passives, advanced resistor levels, breakdown fields (resistor noise `KF`/`AF`/`EF`/`LF`/`WF`
and `noisy` are ported for `.noise`, [NOISE.md](NOISE.md)), unsupported aliases (such as `tc`), advanced setters on literal-only cards,
global geometry scaling and general unported semantics remain gaps. The normal
bounded path rejects invalid thermal/geometry values even where C's
perturbation path permits them.

Owner-crate tests check formulas, precedence, duplicates, signs, missing/invalid
geometry, overflow, immutable recipes and atomic failures. Production tests compare
literal/model DC, source sweeps and complex AC; a contextual RC step matches its
analytic BDF transient bound. Bitwise ordinary-versus-replay tests vary model and instance precedence,
geometry, polynomial/exponential TC, explicit TEMP and DTEMP, and per-point
geometry/turns setters. Live C tests check effective values and DC/AC responses
at default and nondefault TEMP/TNOM without changing committed goldens. They also
exposed an existing logarithmic-AC integer-span roundoff bug; a bounded arithmetic
snap preserves the endpoint, with a regression for integer/noninteger spans and
no change to value-comparison tolerances. See
[VERIFICATION.md](VERIFICATION.md#bounded-passive-elaboration-verification) for
commands, counts and bounds.

## Source-prepared initialization

A native deck keeps ngspice's existing `.ic`, `.nodeset`, instance
`ic=` and UIC behavior. An explicitly prepared source deck can additionally
carry the preprocessing library's `passive_ic::InitialConditions` sidecar.
After the electrical artifacts have been materialized, parse the primary
artifact with `Parser::parse_file_with_output`, then call
`PreparedInitialConditions::attach(&mut parsed.netlist, bundle_directory,
initial_conditions, analyses)`, supplying the selected native job kinds
and UIC flags before elaborating the circuit. The artifact directory is the
emitted bundle root, not the original source directory. Do not discard the typed sidecar or
serialize it into native IC syntax. The native writer refuses to serialize
a deck with attached prepared initialization metadata.

The binder validates exact emitted lexical device identities and resolves
the policy once per elaborated occurrence and enclosing analysis using
its parameter/function scope and the producer's normalized scalar evaluator.
The resolved equations are independent of the source language:

- Hard capacitor bias uses an exact differential voltage source with a
  reserved current row. In transient and AC the row enforces zero current,
  and ordinary capacitor charge remains the only physical storage.
- Soft capacitor/node bias uses the specified finite Norton conductance.
- Hard/soft inductor bias reuses the oriented branch current and enforces
  `i - g*v = target`; ordinary transient flux uses the total terminal current.
- The ordinary bias solve and its continuation attempts retain the
  constraints. Small-signal and transient steps do not.
- UIC bypasses bias equations and uses explicit storage initial values
  before differential node hints. Native node-hint row replacement is not
  used for the prepared soft node bias.

Prepared analysis selection is distinct from Newton load mode. A newly
introduced analysis must be present when the policy is bound; a command-only
deck must explicitly select its enclosing analyses too. The original producer
fixture snapshot under `tests/fixtures/passive-ic` exercises the actual
prepared artifacts through the solver, with independent voltage references.

## Prepared piecewise-linear resistors

The prepared artifact's typed PWL resistor declarations supply signed
voltage/current knots, an independent coordinate, optional one-based initial
segment, and the original emitted law/sensor identities. Pass these alongside
prepared source waveforms in `PreparedDeclarations` when constructing a
prepared circuit. The solver validates the declaration and its physical
anchors in the original lexical definition (artifact, scope and name);
each expanded occurrence retains immutable prepared curve material.

A native branch replaces the emitted behavioral law. Voltage-defined curves
stamp I - G V = I0; current-defined curves stamp V - R I = V0. Existing
zero-voltage observation sensors retain their original orientation and names.
Both forms support negative or zero interior slopes and positive extrapolated
end slopes, with no numerical conductance added. At an interior knot the
right-hand segment supplies the deterministic tangent. Small signal and
pole-zero use the converged branch coordinate's tangent.

An advisory initial segment supplies the first junction-mode Newton tangent
for a fresh bias attempt, including the operating point before transient.
It is not a terminal voltage or current constraint. Subsequent loads use the
actual independent coordinate, even when the initial hint was inconsistent;
convergence checks the physical law. Predicted sweep points use the previous
solution rather than reapplying a startup constraint. Candidate elaboration
retains the original source identities and immutable law data.

## Producer-selected polynomial storage (#252)

The explicit prepared model families `pspice_cap` and `pspice_ind` bind the
source producer's differential-storage law. They do not change the ordinary
native C/L models. Model setters are `c,vc1,vc2` or `l,il1,il2` respectively,
plus `tc1,tc2,temp,dtemp,tnom`; each instance requires a signed nonzero finite
scalar value. The model's dimensionless multiplier may make the effective
value zero. The initial-coordinate setter is `ic` (V or A), and instance
temperature and TC setters override the declaration. No artificial
conductance or independent noise source is added.

For an effective base B and differential coefficients a,b, the stored
quantity is S(x)=B(x+a*x²/2+b*x³/3), and its Jacobian is
D(x)=B(1+a*x+b*x²). The current or voltage branch equation differentiates
S, rather than multiplying x by D. DC sees an open capacitor or short
inductor; AC and pole-zero use the local differential at the bias point.
Transient history contains S and its time derivative, and UIC seeds the
nonlinear stored quantity at the selected coordinate. Candidate replacement
rebinds the effective law without changing the branch or history layout.
Physical Taylor assembly lifts the same law exactly through cubic order.

Native positive `scale` multiplies the base and `m` represents parallel
copies: capacitor charge multiplies by m; inductor flux is S(I/m), with
total branch current I. These are native prepared-model extensions, not a
claim that PSpice accepts these instance keywords. Nonlinear mutual
inductance is rejected until a coupled energy law is available.
Source-specific non-UIC initial bias semantics require a separate producer
policy; the native instance IC alone does not impose a DC clamp.


## Explicit physical passive networks

The native normalized C/L grammar accepts explicit per-copy series resistance and inductance (`pn_rseries`, `pn_lseries`), resistance across the series inductance (`pn_rseries_shunt`), resistance across the main storage (`pn_rstorage_shunt`), and external parallel resistance and capacitance (`pn_rparallel`, `pn_cparallel`). Omitted or zero optional parasitics mean no corresponding element. Ordinary `m`, `temp` and `ic` are resolved per occurrence; multiplicity scales the physical parallel copies. A voltage source supports signed `pn_rseries`, `pn_cparallel` across its output, `pn_cseries_shunt` across its series resistance, and `pn_temp`. The original source excitation and winding identities remain available for sweeps and mutual inductance.

For source-prepared bias, `pn_bias_op`, `pn_bias_dc`, `pn_bias_ac` and `pn_bias_tran` specify the per-copy constraint conductance; presence is significant (zero is a hard constraint). Analysis-dependent damping may be placed across the series inductance, storage or terminal nodes using `pn_gseries_{dc,ac,tran}`, `pn_gstorage_{dc,ac,tran}` or `pn_gterminal_{dc,ac,tran}` and their `_gmin` variants. The latter multiply configured junction gmin rather than continuation conductance. The corresponding `pn_gseries_noise`, `pn_gstorage_noise` and `pn_gterminal_noise` flags select physical thermal noise. The enclosing analysis selects damping even during UIC, independently of bias constraints.

An independent terminal bias uses `pn_terminal_bias_{op,dc,ac,tran}` instead of `pn_bias_*`. It specifies an aggregate, not per-copy, conductance and uses aggregate inductor `ic` current. During the selected bias solve the primary capacitor is open, or the primary winding carries its prescribed current; the alternate terminal constraint replaces the ordinary shunt laws. Outside that phase the original physical network is restored. `pn_exact_ic=1` separately selects exact prescribed primary storage at non-UIC transient startup. `pn_parallel_ic` independently sets the initial voltage of the external parallel capacitance, when present. Without these selectors, ordinary native startup history is unchanged. The reported initial bias point and the physical right-limit storage state need not coincide, including with derivative-constrained systems.

Reporting and controlled-device references are independent of electrical loading. With explicit reporting, `pn_report_{dc,ac,tran}` selects the primary (0) or external terminal (1) current; omitted selectors default to terminal. The optional `_g` term multiplies external differential voltage, and `_rparallel_factor` multiplies the actual aggregate external parallel conductance before multiplying that voltage. These signed terms are projections of solved physical rows, never injected currents. `pn_control` selects the physical control current (0 primary, 1 terminal). The generated ideal port-current probe is not an independent excitation. An intrinsic branch current remains separately available as `i(<name>#intrinsic)` when a reported projection replaces its ordinary output name; the physical port probe is `i(v<name>#port)`.

Source-language omission rules, source-dependent parasitic placement, and reporting/control choices belong to preprocessing, not to the native solver. The independent synthetic fixtures in `tests/fixtures/passive_network` cover the observed distinctions without embedding a vendor-specific compatibility mode.

### Signed physical coefficients

Explicit native networks accept finite signed primary and auxiliary storage,
series and shunt resistance, and analysis-dependent conductance. Parallel-copy
scaling is algebraic: C times m, L and R divided by m, conductance times m.
Zero multiplicity removes electrical objects. Nonzero coefficients may not
underflow to zero or overflow; a resistor must have a finite reciprocal.
A signed constitutive equation is not a claim of passivity: a coupled
inductive system with negative self inductance is rejected by the existing
positive-semidefinite requirement. Classical literal/model validation
remains separate from the signed physical kernel.

Thermal-generator law is independent of the electrical coefficient.
`pn_rparallel_noise_abs=1` selects magnitude of the signed external
parallel conductance in its thermal generator. Other explicit network
losses require nonnegative thermal conductance; a negative generator is
diagnosed when noise is requested, not when electrical analysis is performed.
`pn_noiseless=1` removes all network noise generators without removing
any electrical stamps. Both are occurrence-bound Boolean selectors.
The voltage-source output-impedance thermal generator continues to use
conductance magnitude. Ordinary resistor noise retains its own reference
signed-density arithmetic.
