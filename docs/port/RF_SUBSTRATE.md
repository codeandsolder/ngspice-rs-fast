# Prepared substrate-aware passive RF networks

## Ownership and interface

Architecture campaign #173 defines the numerical boundary for passive RF
extraction. Geometry, material preparation and imported dispersion tables are
offline operations. The solver receives validated SI-valued coefficients and
physical port pairs, not substrate syntax or a geometry parser.

The implementation is in `analysis::rf_network` and
`devices::rf_passive`. It composes with ordinary MNA devices and the existing
AC, SP and Touchstone paths. The T/O/U/Y transmission-line implementations are
unchanged.

- `LineMode::new(ModeParameters)` validates a constitutive mode.
- `PreparedLine::single` or `PreparedLine::modal` adds length and a declared
  frequency band. The latter takes up to 16 modes and a row-major orthogonal
  transform whose rows are modal vectors.
- `propagation`, `rlgc_matrices`, `relations` and `scattering` expose
  numerical results without topology binding.
- `RfLine::new(name, Arc<PreparedLine>, ports)` attaches the distributed
  model to a circuit. Ports are all near-end conductor pairs followed by all
  far-end conductor pairs; each pair explicitly specifies its return node.
- `PreparedLine::realize(LadderSpec)` prepares a causal finite-order network.
  `bind_realization` attaches it to the native RLC solver. A prepared
  realization can be reused in AC, SP and transient analyses.
- `PassiveRealization::lumped` accepts an externally extracted positive RLC
  graph. Node zero is the common internal reference; the following nodes
  are ports, then internal nodes. `elements` and `internal_node_count`
  expose graph data for inspection and transport.

No new netlist dialect or cross-crate dependency is required. External
extraction can produce the same typed data without using the built-in
reference geometry formulas.

## Constitutive equations

At Laplace frequency s, one mode has per-metre series impedance and shunt
admittance

    Z(s,T) = R(T) + s L + k(T) sqrt(s/pi)
    Y(s)   = G + s C + s DeltaC/(1+s tau)

Here L and C are positive, R, G, k and DeltaC are nonnegative, and tau is
positive when DeltaC is nonzero. Frequencies are Hz, temperatures Celsius,
lengths metres, and electrical coefficients SI per metre. R scales by
the linear resistivity factor 1+alpha*(T-Tref); k scales by its square root.
The material domain requires this factor to be finite and positive and T
above absolute zero. The square-root term is the causal skin-effect
asymptote, not an exact cylindrical Bessel-function transition.
A loss-tangent target at a single reference frequency can determine
DeltaC with a chosen tau; it does not imply constant broadband loss tangent.

Propagation is gamma=sqrt(Z Y), with nonnegative real and imaginary parts
on the positive-frequency axis. The characteristic impedance is sqrt(Z/Y)
where representable; zero shunt admittance does not prevent evaluation of
the port relation.

For a real orthogonal modal basis U, the physical matrices are
U^T diag(Z) U and U^T diag(Y) U. Voltages and currents transform in the same
basis, preserving instantaneous power. This supports unequal modal
impedances, velocities and loss. Noncommuting general RLGC matrices are not
silently diagonalized by an approximate transform.

Optional ordered, positive L/C samples (`ModeDispersion`) are interpolated
in frequency within their declared domain. Such empirically dispersive
models support AC/SP only: a sampled table is not a causal Laplace-domain
material law. DC uses the explicitly supplied static coefficients. The
shared evaluation cache is bounded to one exact frequency/temperature key
and returns immutable snapshots; a changed table invalidates the cache.

## Exact numerical port equations

For total series impedance z=length*Z and shunt admittance y=length*Y, let
x=sqrt(z y), c=(1+exp(-x))/2 and q=(1-exp(-x))/x, with q(0)=1.
For the two modal ports, currents point into the network:

    c (Inear + Ifar) = y q/2 (Vnear + Vfar)
    c (Vnear - Vfar) = z q/2 (Inear - Ifar)

These relations use bounded exponentials rather than growing ABCD
hyperbolic functions or divisions by sinh(x). A near-zero Taylor expansion,
half-angle identities and expm1 preserve precision around DC and resonances.
Each equation is normalized by its largest coefficient component.
Physical port currents are explicit MNA unknowns; their signed terminal
incidences enforce KCL even for floating/shared return nodes.

At zero length the equations impose equal voltage and opposite current
exactly. At DC a lossless line is likewise an exact through constraint;
a resistive line keeps its physical series resistance. Ideal resonances do
not require an artificial shunt. Scattering with equal real reference
impedance is computed from modal reflection eigenvalues without converting
through a potentially singular Y matrix. Actual circuit SP uses the normal
port and MNA machinery, including heterogeneous port references.

Nonfinite coefficients or unrepresentable results are rejected. Ordinary
circuit singularities (e.g. conflicting ideal sources) still belong to
the solver's usual diagnostics.

## Offline reference geometry

These are quasi-static or empirical preparation models, not full-wave
field solvers. The caller declares an electrical validity band; geometry
checks additionally enforce the stated dimensionless domains.

| Geometry | Model and domain |
|---|---|
| Coax | Homogeneous TEM, ordered positive radii, relative permittivity >=1; conductivity-based DC cross sections and thick-metal skin asymptote |
| Microstrip | Hammerstad–Jensen, W/h 0.01–100, t/h <=0.1 and t<=W/2; finite-thickness width correction |
| Grounded CPW | Conductor-backed conformal map, zero thickness; W/h 0.01–20, gap/h 0.001–20; AGM elliptic integrals |
| Coupled microstrip | Kirschning–Jansen symmetric thin-strip even/odd modes; W/h and gap/h 0.1–10, relative permittivity <=18 |
| Open microstrip | Effective-length fringing capacitance (0.412 formula), W/h 0.1–10; not a radiation model |
| Right-angle corner | Qucs lumped equivalent, thin strip; W/h 0.2–6, relative permittivity 2.36–10.4, f*h <=12 MHz*m; positive-element subset |
| Width step | Qucs impedance-step equivalent, thin strip; width ratio 1.5–3.5, relative permittivity <=10, f*h <=12 MHz*m; positive-element subset |
| Tee | Leading-order Qucs reference-plane and susceptance model; thin strip, each W/h 0.1–10, relative permittivity <=18; each electrical arm length <=0.1 rad and f*h/(4e5*Zarm)<=0.05 |

For planar models substrate relative permittivity is finite and >=1
(subject to the individual limits). Other conductor/return geometries can
supply extracted per-metre coefficients. Surface roughness, radiation and
higher-order mode physics must be supplied by extraction within the chosen
band, rather than guessed from the quasi-static geometry.

The tee approximation includes the first-order series L and split shunt C
of each physical arm. Negative deembedded arm lengths require explicit
positive reference-plane extensions. A negative deembedded central C may
be combined with positive arm capacitances at the same node, but the
result must be nonnegative. Corner and step coefficients must also be
nonnegative; no coefficient is clipped to hide an out-of-domain geometry.
An open stub is a prepared line terminated by its extracted capacitance.

Reference equations:
- [Qucs technical documentation: microstrip](https://qucs.github.io/tech/node75.html),
  [coupled lines](https://qucs.github.io/tech/node77.html),
  [width step](https://qucs.github.io/tech/node80.html),
  [tee](https://qucs.github.io/tech/node81.html).
- [Qucsator reference implementation, pinned revision](https://github.com/Qucs/qucsator/tree/e995f9acc71a8c7319286944e4a1692318b9dd80/src/components/microstrip).
- [scikit-rf microstrip](https://scikit-rf.readthedocs.io/en/latest/api/media/generated/skrf.media.mline.MLine.html)
  and [CPW](https://scikit-rf.readthedocs.io/en/latest/api/media/generated/skrf.media.cpw.CPW.html).

## Passive transient realization

A distributed analytic line is not stamped as a frequency-dependent
resistor during transient integration. Its finite-order approximation is a
cascade of symmetric pi cells, containing native series R/L and shunt G/C.
The dielectric relaxation becomes a positive series RC shunt. The
square-root impedance is approximated by positive Foster RL branches
using logarithmic quadrature of

    sqrt(s) = (1/pi) integral_0^infinity s/(s+t) t^(-1/2) dt.

Candidate spatial section counts are powers of two up to the requested
limit (at most 256); each mode tries 0,4,8,16,32,64 skin poles. The fitter
selects the smallest element graph among candidates passing the specified
absolute modal scattering-eigenvalue error. The bounded validation grid
contains DC, linear and logarithmic samples; the linear grid density also
depends on electrical length (at least 16 intervals per full phase
rotation, subject to the implementation's resource limit). A report
contains section count, total skin poles per section, sample count and
worst observed error. This is a sampled error check, not a mathematical
continuous-frequency certificate. Causality and passivity follow instead
from the positive RLC graph and real power-conserving port transformer.

The fitting band must be inside the line's declared band. The graph is
fixed at the preparation temperature; changed temperature requires
re-preparation. Native transient analysis uses the existing integration,
state and timestep machinery. Transient accuracy is meaningful only for
excitation within the fitted band. Direct analytic distributed stamping
and tabulated dispersion are deliberately not offered as transient
models; extracted passive rational equivalents may be bound separately.

Insertion is atomic, checks aggregate device/node/state limits and
terminal validity, and rejects naming collisions, floating internal DC
components and inductive loops. The existing circuit topology machinery
owns all generated internal nodes and reactive state. Native RLC loss
elements participate in their normal noise handling; the distributed
analytic model has no separate RF thermal-noise implementation here.

## Regression evidence

`tests/rf_substrate_networks.rs` exercises:
- exact delay at resonances, zero-length/DC limits, finite extreme
  reference impedances, reciprocity and power conservation;
- independent microstrip and CPW cross-section references from scikit-rf
  2.2.0 and coupled-strip reference values from the pinned Qucsator C++
  implementation; sampled CPW dispersion at 1 MHz through 10 GHz;
- prepared models bound to actual AC and SP solves, unequal-mode
  four-port native ladder comparison, passive discontinuities and filter
  composition, explicit rejection of nonphysical geometry;
- lossy ladder error checks and native time-domain propagation,
  context/cache correctness and transactional circuit insertion;
- a measured ring-slot antenna Touchstone load composed with a matched
  prepared feed. Three measured complex samples are from scikit-rf 2.2.0
  [ring slot measured.s1p](https://github.com/scikit-rf/scikit-rf/blob/v2.2.0/skrf/data/ring%20slot%20measured.s1p);
  the DC open is a separately declared synthetic bias condition.

The lossy ladder regression selected four spatial sections and 32 skin
poles per section (284 elements), with worst sampled modal scattering
error 0.002774440512064803 over 384 frequencies for a 0.005 requested
budget. These values describe the test geometry and material, not a
universal fitting guarantee.
