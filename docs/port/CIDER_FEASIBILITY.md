# Numerical semiconductor devices (CIDER feasibility, #188)

## Decision and scope

Proceed with an optional native 1D numerical-device kernel, starting with
uniform-material Poisson/drift-diffusion, then an explicitly scoped model-card
subset and circuit integration. Do not add a production dependency or claim
general numerical-device parity from this experiment. Multidomain 2D/3D,
additional transport laws and calibrated process models require independent
acceptance work. For advanced TCAD, an external specialist such as DEVSIM
(Apache-2.0) or Genius (GPLv3) remains useful, particularly as an independent
reference. No mandatory external solver, scripting runtime or legacy ABI is
needed for native numerical devices.

## Upstream inventory

ngspice 47 CIDER registers NUMD (1D and 2D numerical diode, also resistor),
NBJT (1D and 2D numerical bipolar, including a 2D JFET example), and NUMOS
(2D MOS). These supply OP/DC, transient, AC and PZ hooks but not noise or
SOA hooks. The 1D core solves Poisson and electron/hole continuity; material
models cover SRH, optional Auger and avalanche, mobility, bandgap narrowing,
and lifetime dependence. 2D additionally supports multiple semiconductor
and dielectric regions, electrodes, boundary/contact rules and rectangular
meshes. Uniform, linear, Gaussian, erfc, exponential and imported doping
profiles appear in the input vocabulary. There is no general distributed
thermal PDE in the audited 1D core. Device temperature dependence is distinct
from a self-heating PDE. CIDER uses modified BSD terms; KLU has its own
license. Retain applicable notices for any subsequent port.

Sources: upstream [device list](https://github.com/imr/ngspice/blob/master/DEVICES),
[NEWS](https://github.com/imr/ngspice/blob/master/NEWS) (46 added KLU for
CIDER DC/OP/transient), and
[manual](https://nmg.gitlab.io/ngspice-manual/mixed-levelsimulation_ngspicewithtcad/cider.html).
The prototype is independent Rust, not translated C source.

Observed reference runs on ngspice 47 (installed CIDER-enabled binary,
Acer Ryzen 4700U, concurrent load; elapsed time is descriptive):

| Upstream example | Nodes | Output rows | Internal iterations | Wall time / peak RSS |
|---|---:|---:|---:|---:|
| diode/diode.cir | 201 | 51 AC + 1 OP | 113 | 0.69 s / 7080 KiB |
| diode/diotran.cir | 201 | 1022 | 2366 | 3.69 s / 7268 KiB |
| resistor/sires.cir | 101 | 101 | 406 | 0.05 s / 7068 KiB |
| resistor/gaasres.cir | 101 | 112 | 224 | 0.05 s / 6884 KiB |
| jfet/jfet.cir | 720 | 21 | 98 | 17.48 s / 15160 KiB |
| mos/pass.cir | 557 | 33 | 149 | 1.27 s / 9784 KiB |
| diode/pd1.cir | 442 | 1 OP | 30 | 2.70 s / 11356 KiB |

The PIN continuation transient exceeded a 60 s experimental timeout.
Two BJT PZ examples reached their 201-trial limit despite a successful
process exit. A process status alone does not validate numerical results.
The ordinary pinned reference build disables CIDER; production CIDER
conformance should establish a separately pinned CIDER-enabled build.

## Reproducible experiment

`experiments/cider-prototype` is a dependency-free, standalone Cargo
workspace. It has an immutable nonuniform mesh and signed SI doping;
accepted and trial state share the exact prepared geometry. Input geometry,
time step, temperature, nonfinite Newton residuals, and mismatched states
are checked. A fixed uniform silicon material and two ideal ohmic contacts
are intentional experiment limits. The area is 1e-6 m², length 2 µm in the
abrupt PN fixture, net doping ±1e22 m⁻³ with an intrinsic central node,
300 K reference, mobility 0.1/0.04 m²/(V s), SRH lifetime 1 µs. Boltzmann
statistics, no avalanche, no bandgap narrowing or field-dependent mobility.

From the repository root, with the managed compiler wrapper and an isolated
build directory (choose an appropriate path on the build volume):

```sh
cargo +nightly test --locked --release --manifest-path experiments/cider-prototype/Cargo.toml --target-dir "$build/target" --config "build.build-dir='$build/target'"
cargo +nightly clippy --locked --manifest-path experiments/cider-prototype/Cargo.toml --all-targets --target-dir "$build/target" --config "build.build-dir='$build/target'" -- -D warnings
cargo +nightly run --locked --release --manifest-path experiments/cider-prototype/Cargo.toml --target-dir "$build/target" --config "build.build-dir='$build/target'" -- 201 dc
```

Other CLI modes: `201 dc clustered`, `201 tran 0.5` (step in ps),
`201 thermal`, `201 stamp`, `201 ac 0.55`, `201 uniform`,
`201 ac-tran 0.55 1e9 128 8 1e-5`. These are research commands, not
production netlist syntax. Unknown modes are rejected.

The committed original controlled reference decks and 17-digit CIDER DC/AC
samples are under the experiment's `reference` directory. To regenerate,
copy each deck into an isolated scratch directory, run `ngspice -b`, and
convert voltage-source currents to device currents by negating them.
For AC, replace the template's voltage source DC bias (0.55 V) with
-0.3, 0, 0.3 and 0.55 V and collect the real/imaginary output for each
bias. DC has 39 points, AC has 564 (141 per bias). The decks explicitly
set both majority and minority mobility and nominal temperature;
leaving default minority mobilities changes the forward current.

## Numerical method and evidence

Nonuniform finite volumes with Scharfetter–Gummel flux, analytic forward
differentiation of the residual, row-scaled Newton and pivoted block
tridiagonal elimination. Positive carrier densities follow from log
variables. Separate majority-contact quasi-Fermi gauges and expm1 edge
flux avoid subtraction of nearly equal minority-carrier terms. Bernoulli
evaluation uses small-argument and large-argument stable forms. Terminal
current includes displacement current; heat generation uses transport and
recombination dissipation rather than equating all transient VI to heat.

Original controlled experiments, 201-node PN:

- 38 nonzero DC points, -0.3 to +0.65 V, worst CIDER relative discrepancy
  3.154e-8, median 7.21e-11; equilibrium current zero to numerical
  precision. At -0.3 V current is about -54.85 pA; at +0.65 V, 0.21039 A.
- Mesh error against 801 uniform nodes: 101 uniform 1.857% maximum current
  error vs 101 clustered 0.0751%; 201 uniform 0.0337% vs 201 clustered
  0.00682%. Analytic depletion estimate Vbi=0.73522 V, width=0.43606 µm
  at zero bias; peak field 3.3721 MV/m vs numerical 3.1744 MV/m, expected
  difference from diffuse-carrier effects.
- Backward Euler reverse recovery (0.55 to -0.3 V over 0.2 ns, simulated
  to 1 ns) at 10/2/0.5 ps yields overlapping C trace RMS error
  0.04883/0.01231/0.002947 A. Peak current at 0.5 ps -1.8972 A.
  Strict C BE stops at 0.553 ns; relaxed trapezoidal SPARSE/KLU reaches
  1 ns. Do not compare different integration schemes as exact goldens.
- Lumped isothermal self-heating demonstration, Rth=500 K/W, Cth=2 nJ/K:
  peak 317.481 K, end 303.985 K, maximum heat-balance residual
  2.019e-10 W; DC dissipation vs VI within 2.50e-14 W.
  This is not a distributed thermal or thermoelectric model.
- Condensed terminal DC/transient derivative vs perturbation <=2.973e-10
  relative in the original sweep. Ten 1 kΩ series-R circuit operating
  points agreed with CIDER to 0.693 nV maximum diode-voltage difference.
- Reusable dynamic elimination of J+sM, including carrier storage and
  displacement current: four-bias C AC sweep worst relative discrepancy
  8.08e-5 at low frequency/near equilibrium; at +0.55 V, 3.80e-12.
  The low-frequency C conductance cancellation was independently checked
  by Richardson-extrapolated DC perturbations (agreement with native DC
  derivative <=5.5e-11 relative).
- Homogeneous analytic Y(s)=A/L * [q(mu_n n + mu_p p) + s epsilon],
  1692 comparisons, worst relative 2.52e-12. At real s=1/dt dynamic
  derivative agrees with independently assembled BE derivative <=3.06e-13
  relative (12 bias/step cases).
- Six nonlinear sine transients vs exact discrete BE small-signal
  response, max relative 1.81e-8. Continuous and discrete derivatives
  are intentionally distinguished. The 100 THz numerical sweep probes
  limiting behavior, not physical drift-diffusion validity at 100 THz.

Committed tests enforce the controlled C DC and AC results, independent
homogeneous admittance, DC and transient terminal coupling, temperature
balance, state isolation and input checks. The low-frequency C comparison
tolerance is separate from the tighter analytic and BE comparisons.

## General coupling architecture and follow-up

Immutable mesh and topology, typed regions/materials/contacts/terminals;
physical geometry in SI and scaling internal to the equation family.
Unknown families declare residual, Jacobian and storage/mass derivatives
with boundary conditions and conservation at interfaces. Separate accepted
history, mutable trial workspace and immutable prepared structure. For
partitioned coupling with internal variables x and terminal variables v,
condense the residual F(x,v,h)=0 to circuit terminal I(x,v,h) using
I_v - I_x F_x^-1 F_v. Factor once, solve all terminal RHS with the existing
sparse batched machinery; for AC/PZ use J+sM, including voltage/storage
derivatives and charge/displacement contributions. Monolithic assembly may
be appropriate for strongly coupled devices. Internal block elimination is
a 1D specialization, not a substitute for the general sparse solver.

The current prepared-device contract and engine accepted/trial lifecycle
provide the relevant MNA topology, residual and state ownership primitives.
A future numerical-device interface must additionally define prepared
mesh identity, internal trial convergence and LTE, terminal current and
charge, nonlinear history and optional thermal nodes; do not force all
interior PDE unknowns into the compact-model interface. Independent device
trials may run in parallel with deterministic circuit stamps.

Recommended staged tasks:

1. Production-grade typed 1D mesh/material/contact and residual/storage
   kernel, reusable numerical workspace, adaptive integration/LTE,
   validated scaling and pivot fallback.
2. Circuit integration with transactional trial state and terminal Jacobian;
   explicit supported-card parser and structured unsupported errors (never
   compact-model substitution). Pinned CIDER-enabled reference.
3. AC/PZ dynamic condensation and observable state on nonuniform meshes;
   independent conservation, analytic and C comparisons.
4. Multidomain 2D device kernels and interface/boundary conditions, followed
   by BJT/MOS/JFET and justified additional transport/material laws.

The standalone prototype leaves ordinary compact-model execution unchanged.
It does not implement the production parser, general multi-domain mesh,
adaptive integration, arbitrary material physics, full CIDER parity or
thermal PDE.
